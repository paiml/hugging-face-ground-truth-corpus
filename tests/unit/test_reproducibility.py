"""Tests for reproducible training utilities."""

from __future__ import annotations

import random

import pytest

from hf_gtc.training.reproducibility import (
    VALID_CHECKSUM_TYPES,
    VALID_DETERMINISM_LEVELS,
    VALID_SEED_COMPONENTS,
    ChecksumType,
    DeterminismLevel,
    EnvironmentInfo,
    ReproducibilityConfig,
    ReproducibilityStats,
    RNGState,
    SeedComponent,
    SeedConfig,
    compute_config_hash,
    compute_environment_hash,
    create_environment_info,
    create_reproducibility_config,
    create_reproducibility_stats,
    create_rng_state,
    create_seed_config,
    format_environment_info,
    format_reproducibility_stats,
    get_checksum_type,
    get_determinism_level,
    get_recommended_reproducibility_config,
    get_rng_state,
    get_seed_component,
    list_checksum_types,
    list_determinism_levels,
    list_seed_components,
    set_all_seeds,
    set_rng_state,
    validate_environment_info,
    validate_reproducibility_config,
    validate_reproducibility_stats,
    validate_rng_state,
    validate_seed_config,
    verify_reproducibility,
)


class TestSeedComponent:
    """Tests for SeedComponent enum."""

    def test_python_value(self) -> None:
        """Test PYTHON component value."""
        assert SeedComponent.PYTHON.value == "python"

    def test_numpy_value(self) -> None:
        """Test NUMPY component value."""
        assert SeedComponent.NUMPY.value == "numpy"

    def test_torch_value(self) -> None:
        """Test TORCH component value."""
        assert SeedComponent.TORCH.value == "torch"

    def test_cuda_value(self) -> None:
        """Test CUDA component value."""
        assert SeedComponent.CUDA.value == "cuda"

    def test_transformers_value(self) -> None:
        """Test TRANSFORMERS component value."""
        assert SeedComponent.TRANSFORMERS.value == "transformers"

    def test_all_components_in_valid_set(self) -> None:
        """Test all enum values are in VALID_SEED_COMPONENTS."""
        for component in SeedComponent:
            assert component.value in VALID_SEED_COMPONENTS


class TestDeterminismLevel:
    """Tests for DeterminismLevel enum."""

    def test_none_value(self) -> None:
        """Test NONE level value."""
        assert DeterminismLevel.NONE.value == "none"

    def test_partial_value(self) -> None:
        """Test PARTIAL level value."""
        assert DeterminismLevel.PARTIAL.value == "partial"

    def test_full_value(self) -> None:
        """Test FULL level value."""
        assert DeterminismLevel.FULL.value == "full"

    def test_all_levels_in_valid_set(self) -> None:
        """Test all enum values are in VALID_DETERMINISM_LEVELS."""
        for level in DeterminismLevel:
            assert level.value in VALID_DETERMINISM_LEVELS


class TestChecksumType:
    """Tests for ChecksumType enum."""

    def test_md5_value(self) -> None:
        """Test MD5 type value."""
        assert ChecksumType.MD5.value == "md5"

    def test_sha256_value(self) -> None:
        """Test SHA256 type value."""
        assert ChecksumType.SHA256.value == "sha256"

    def test_xxhash_value(self) -> None:
        """Test XXHASH type value."""
        assert ChecksumType.XXHASH.value == "xxhash"

    def test_all_types_in_valid_set(self) -> None:
        """Test all enum values are in VALID_CHECKSUM_TYPES."""
        for checksum_type in ChecksumType:
            assert checksum_type.value in VALID_CHECKSUM_TYPES


class TestSeedConfig:
    """Tests for SeedConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating seed config."""
        config = SeedConfig(
            seed=42,
            components=(SeedComponent.PYTHON, SeedComponent.NUMPY),
            deterministic_algorithms=True,
        )
        assert config.seed == 42
        assert len(config.components) == 2
        assert config.deterministic_algorithms is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = SeedConfig(
            seed=42,
            components=(SeedComponent.PYTHON,),
            deterministic_algorithms=False,
        )
        with pytest.raises(AttributeError):
            config.seed = 100  # type: ignore[misc]


class TestReproducibilityConfig:
    """Tests for ReproducibilityConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating reproducibility config."""
        seed_cfg = SeedConfig(42, (SeedComponent.PYTHON,), False)
        config = ReproducibilityConfig(
            seed_config=seed_cfg,
            determinism_level=DeterminismLevel.PARTIAL,
            log_environment=True,
            save_rng_state=True,
        )
        assert config.seed_config.seed == 42
        assert config.determinism_level == DeterminismLevel.PARTIAL
        assert config.log_environment is True
        assert config.save_rng_state is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        seed_cfg = SeedConfig(42, (SeedComponent.PYTHON,), False)
        config = ReproducibilityConfig(
            seed_config=seed_cfg,
            determinism_level=DeterminismLevel.PARTIAL,
            log_environment=True,
            save_rng_state=True,
        )
        with pytest.raises(AttributeError):
            config.log_environment = False  # type: ignore[misc]


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo dataclass."""

    def test_create_info(self) -> None:
        """Test creating environment info."""
        info = EnvironmentInfo(
            python_version="3.11.5",
            torch_version="2.1.0",
            cuda_version="12.1",
            transformers_version="4.35.0",
        )
        assert info.python_version == "3.11.5"
        assert info.torch_version == "2.1.0"
        assert info.cuda_version == "12.1"
        assert info.transformers_version == "4.35.0"

    def test_optional_versions(self) -> None:
        """Test optional version fields."""
        info = EnvironmentInfo(
            python_version="3.11.5",
            torch_version=None,
            cuda_version=None,
            transformers_version=None,
        )
        assert info.torch_version is None
        assert info.cuda_version is None
        assert info.transformers_version is None

    def test_frozen(self) -> None:
        """Test info is immutable."""
        info = EnvironmentInfo("3.11.5", None, None, None)
        with pytest.raises(AttributeError):
            info.python_version = "3.12.0"  # type: ignore[misc]


class TestRNGState:
    """Tests for RNGState dataclass."""

    def test_create_state(self) -> None:
        """Test creating RNG state."""
        state = RNGState(
            python_state=b"python",
            numpy_state=b"numpy",
            torch_state=b"torch",
            cuda_state=b"cuda",
        )
        assert state.python_state == b"python"
        assert state.numpy_state == b"numpy"
        assert state.torch_state == b"torch"
        assert state.cuda_state == b"cuda"

    def test_partial_state(self) -> None:
        """Test partial RNG state."""
        state = RNGState(
            python_state=b"python",
            numpy_state=None,
            torch_state=None,
            cuda_state=None,
        )
        assert state.python_state == b"python"
        assert state.numpy_state is None

    def test_frozen(self) -> None:
        """Test state is immutable."""
        state = RNGState(b"python", None, None, None)
        with pytest.raises(AttributeError):
            state.python_state = b"new"  # type: ignore[misc]


class TestReproducibilityStats:
    """Tests for ReproducibilityStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating stats."""
        stats = ReproducibilityStats(
            seed=42,
            determinism_level=DeterminismLevel.FULL,
            env_hash="abc123",
            config_hash="def456",
        )
        assert stats.seed == 42
        assert stats.determinism_level == DeterminismLevel.FULL
        assert stats.env_hash == "abc123"
        assert stats.config_hash == "def456"

    def test_frozen(self) -> None:
        """Test stats is immutable."""
        stats = ReproducibilityStats(42, DeterminismLevel.FULL, "abc", "def")
        with pytest.raises(AttributeError):
            stats.seed = 100  # type: ignore[misc]


class TestValidateSeedConfig:
    """Tests for validate_seed_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = SeedConfig(42, (SeedComponent.PYTHON,), False)
        validate_seed_config(config)  # Should not raise

    def test_none_config(self) -> None:
        """Test None config."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_seed_config(None)  # type: ignore[arg-type]

    def test_negative_seed(self) -> None:
        """Test negative seed."""
        config = SeedConfig(-1, (SeedComponent.PYTHON,), False)
        with pytest.raises(ValueError, match="seed must be non-negative"):
            validate_seed_config(config)

    def test_empty_components(self) -> None:
        """Test empty components."""
        config = SeedConfig(42, (), False)
        with pytest.raises(ValueError, match="components cannot be empty"):
            validate_seed_config(config)


class TestValidateReproducibilityConfig:
    """Tests for validate_reproducibility_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        seed_cfg = SeedConfig(42, (SeedComponent.PYTHON,), False)
        config = ReproducibilityConfig(seed_cfg, DeterminismLevel.PARTIAL, True, True)
        validate_reproducibility_config(config)  # Should not raise

    def test_none_config(self) -> None:
        """Test None config."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_reproducibility_config(None)  # type: ignore[arg-type]

    def test_invalid_seed_config(self) -> None:
        """Test invalid nested seed config."""
        seed_cfg = SeedConfig(-1, (SeedComponent.PYTHON,), False)
        config = ReproducibilityConfig(seed_cfg, DeterminismLevel.PARTIAL, True, True)
        with pytest.raises(ValueError, match="seed must be non-negative"):
            validate_reproducibility_config(config)


class TestValidateEnvironmentInfo:
    """Tests for validate_environment_info function."""

    def test_valid_info(self) -> None:
        """Test validating valid info."""
        info = EnvironmentInfo("3.11.5", "2.1.0", "12.1", "4.35.0")
        validate_environment_info(info)  # Should not raise

    def test_none_info(self) -> None:
        """Test None info."""
        with pytest.raises(ValueError, match="info cannot be None"):
            validate_environment_info(None)  # type: ignore[arg-type]

    def test_empty_python_version(self) -> None:
        """Test empty python version."""
        info = EnvironmentInfo("", None, None, None)
        with pytest.raises(ValueError, match="python_version cannot be empty"):
            validate_environment_info(info)


class TestValidateRNGState:
    """Tests for validate_rng_state function."""

    def test_valid_state(self) -> None:
        """Test validating valid state."""
        state = RNGState(b"python", None, None, None)
        validate_rng_state(state)  # Should not raise

    def test_none_state(self) -> None:
        """Test None state."""
        with pytest.raises(ValueError, match="state cannot be None"):
            validate_rng_state(None)  # type: ignore[arg-type]

    def test_all_none_states(self) -> None:
        """Test all None states."""
        state = RNGState(None, None, None, None)
        with pytest.raises(ValueError, match="at least one RNG state"):
            validate_rng_state(state)


class TestValidateReproducibilityStats:
    """Tests for validate_reproducibility_stats function."""

    def test_valid_stats(self) -> None:
        """Test validating valid stats."""
        stats = ReproducibilityStats(42, DeterminismLevel.PARTIAL, "abc", "def")
        validate_reproducibility_stats(stats)  # Should not raise

    def test_none_stats(self) -> None:
        """Test None stats."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_reproducibility_stats(None)  # type: ignore[arg-type]

    def test_negative_seed(self) -> None:
        """Test negative seed."""
        stats = ReproducibilityStats(-1, DeterminismLevel.PARTIAL, "abc", "def")
        with pytest.raises(ValueError, match="seed must be non-negative"):
            validate_reproducibility_stats(stats)

    def test_empty_env_hash(self) -> None:
        """Test empty env_hash."""
        stats = ReproducibilityStats(42, DeterminismLevel.PARTIAL, "", "def")
        with pytest.raises(ValueError, match="env_hash cannot be empty"):
            validate_reproducibility_stats(stats)

    def test_empty_config_hash(self) -> None:
        """Test empty config_hash."""
        stats = ReproducibilityStats(42, DeterminismLevel.PARTIAL, "abc", "")
        with pytest.raises(ValueError, match="config_hash cannot be empty"):
            validate_reproducibility_stats(stats)


class TestCreateSeedConfig:
    """Tests for create_seed_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_seed_config()
        assert config.seed == 42
        assert len(config.components) == len(SeedComponent)
        assert config.deterministic_algorithms is False

    def test_custom_seed(self) -> None:
        """Test custom seed."""
        config = create_seed_config(seed=123)
        assert config.seed == 123

    def test_custom_components(self) -> None:
        """Test custom components."""
        config = create_seed_config(components=(SeedComponent.PYTHON,))
        assert len(config.components) == 1
        assert SeedComponent.PYTHON in config.components

    def test_deterministic_algorithms(self) -> None:
        """Test deterministic_algorithms flag."""
        config = create_seed_config(deterministic_algorithms=True)
        assert config.deterministic_algorithms is True

    def test_negative_seed(self) -> None:
        """Test negative seed raises error."""
        with pytest.raises(ValueError, match="seed must be non-negative"):
            create_seed_config(seed=-1)

    def test_empty_components(self) -> None:
        """Test empty components raises error."""
        with pytest.raises(ValueError, match="components cannot be empty"):
            create_seed_config(components=())


class TestCreateReproducibilityConfig:
    """Tests for create_reproducibility_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_reproducibility_config()
        assert config.seed_config.seed == 42
        assert config.determinism_level == DeterminismLevel.PARTIAL
        assert config.log_environment is True
        assert config.save_rng_state is True

    def test_custom_seed_config(self) -> None:
        """Test custom seed config."""
        seed_cfg = create_seed_config(seed=123)
        config = create_reproducibility_config(seed_config=seed_cfg)
        assert config.seed_config.seed == 123

    def test_determinism_level_string(self) -> None:
        """Test determinism level from string."""
        config = create_reproducibility_config(determinism_level="full")
        assert config.determinism_level == DeterminismLevel.FULL

    def test_determinism_level_enum(self) -> None:
        """Test determinism level from enum."""
        config = create_reproducibility_config(determinism_level=DeterminismLevel.NONE)
        assert config.determinism_level == DeterminismLevel.NONE

    def test_invalid_determinism_level(self) -> None:
        """Test invalid determinism level."""
        with pytest.raises(ValueError, match="determinism_level must be one of"):
            create_reproducibility_config(determinism_level="invalid")

    def test_log_environment_false(self) -> None:
        """Test log_environment=False."""
        config = create_reproducibility_config(log_environment=False)
        assert config.log_environment is False


class TestCreateEnvironmentInfo:
    """Tests for create_environment_info function."""

    def test_auto_detect_python(self) -> None:
        """Test auto-detection of Python version."""
        info = create_environment_info()
        assert len(info.python_version) > 0

    def test_custom_versions(self) -> None:
        """Test custom version strings."""
        info = create_environment_info(
            python_version="3.11.5",
            torch_version="2.1.0",
            cuda_version="12.1",
            transformers_version="4.35.0",
        )
        assert info.python_version == "3.11.5"
        assert info.torch_version == "2.1.0"
        assert info.cuda_version == "12.1"
        assert info.transformers_version == "4.35.0"

    def test_partial_versions(self) -> None:
        """Test partial version info."""
        info = create_environment_info(
            python_version="3.11.5",
            torch_version="2.1.0",
        )
        assert info.torch_version == "2.1.0"
        assert info.cuda_version is None


class TestCreateRNGState:
    """Tests for create_rng_state function."""

    def test_python_state(self) -> None:
        """Test creating state with Python state."""
        state = create_rng_state(python_state=b"test")
        assert state.python_state == b"test"

    def test_multiple_states(self) -> None:
        """Test creating state with multiple states."""
        state = create_rng_state(
            python_state=b"py",
            numpy_state=b"np",
        )
        assert state.python_state == b"py"
        assert state.numpy_state == b"np"

    def test_no_states_raises_error(self) -> None:
        """Test no states raises error."""
        with pytest.raises(ValueError, match="at least one RNG state"):
            create_rng_state()


class TestCreateReproducibilityStats:
    """Tests for create_reproducibility_stats function."""

    def test_valid_stats(self) -> None:
        """Test creating valid stats."""
        stats = create_reproducibility_stats(
            seed=42,
            determinism_level=DeterminismLevel.FULL,
            env_hash="abc123",
            config_hash="def456",
        )
        assert stats.seed == 42
        assert stats.env_hash == "abc123"

    def test_empty_env_hash(self) -> None:
        """Test empty env_hash raises error."""
        with pytest.raises(ValueError, match="env_hash cannot be empty"):
            create_reproducibility_stats(env_hash="", config_hash="abc")

    def test_empty_config_hash(self) -> None:
        """Test empty config_hash raises error."""
        with pytest.raises(ValueError, match="config_hash cannot be empty"):
            create_reproducibility_stats(env_hash="abc", config_hash="")


class TestListSeedComponents:
    """Tests for list_seed_components function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        components = list_seed_components()
        assert isinstance(components, list)

    def test_contains_expected_components(self) -> None:
        """Test contains expected components."""
        components = list_seed_components()
        assert "python" in components
        assert "numpy" in components
        assert "torch" in components

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        components = list_seed_components()
        assert components == sorted(components)


class TestListDeterminismLevels:
    """Tests for list_determinism_levels function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        levels = list_determinism_levels()
        assert isinstance(levels, list)

    def test_contains_expected_levels(self) -> None:
        """Test contains expected levels."""
        levels = list_determinism_levels()
        assert "none" in levels
        assert "partial" in levels
        assert "full" in levels

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        levels = list_determinism_levels()
        assert levels == sorted(levels)


class TestListChecksumTypes:
    """Tests for list_checksum_types function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        types = list_checksum_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_checksum_types()
        assert "md5" in types
        assert "sha256" in types
        assert "xxhash" in types

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        types = list_checksum_types()
        assert types == sorted(types)


class TestGetSeedComponent:
    """Tests for get_seed_component function."""

    def test_python(self) -> None:
        """Test getting PYTHON component."""
        assert get_seed_component("python") == SeedComponent.PYTHON

    def test_torch(self) -> None:
        """Test getting TORCH component."""
        assert get_seed_component("torch") == SeedComponent.TORCH

    def test_invalid(self) -> None:
        """Test invalid component."""
        with pytest.raises(ValueError, match="seed_component must be one of"):
            get_seed_component("invalid")


class TestGetDeterminismLevel:
    """Tests for get_determinism_level function."""

    def test_none(self) -> None:
        """Test getting NONE level."""
        assert get_determinism_level("none") == DeterminismLevel.NONE

    def test_full(self) -> None:
        """Test getting FULL level."""
        assert get_determinism_level("full") == DeterminismLevel.FULL

    def test_invalid(self) -> None:
        """Test invalid level."""
        with pytest.raises(ValueError, match="determinism_level must be one of"):
            get_determinism_level("invalid")


class TestGetChecksumType:
    """Tests for get_checksum_type function."""

    def test_md5(self) -> None:
        """Test getting MD5 type."""
        assert get_checksum_type("md5") == ChecksumType.MD5

    def test_sha256(self) -> None:
        """Test getting SHA256 type."""
        assert get_checksum_type("sha256") == ChecksumType.SHA256

    def test_invalid(self) -> None:
        """Test invalid type."""
        with pytest.raises(ValueError, match="checksum_type must be one of"):
            get_checksum_type("invalid")


class TestSetAllSeeds:
    """Tests for set_all_seeds function."""

    def test_python_seed(self) -> None:
        """Test setting Python seed."""
        config = create_seed_config(seed=42, components=(SeedComponent.PYTHON,))
        result = set_all_seeds(config)
        assert result["python"] is True

        # Verify seed was set
        random.seed(42)
        expected = random.random()
        random.seed(42)
        actual = random.random()
        assert expected == actual

    def test_none_config(self) -> None:
        """Test None config raises error."""
        with pytest.raises(ValueError, match="config cannot be None"):
            set_all_seeds(None)  # type: ignore[arg-type]

    def test_multiple_components(self) -> None:
        """Test setting multiple components."""
        config = create_seed_config(
            seed=42,
            components=(SeedComponent.PYTHON, SeedComponent.NUMPY),
        )
        result = set_all_seeds(config)
        assert result["python"] is True
        # numpy may or may not be available
        assert "numpy" in result


class TestGetRNGState:
    """Tests for get_rng_state function."""

    def test_python_state(self) -> None:
        """Test getting Python RNG state."""
        random.seed(42)
        state = get_rng_state(components=(SeedComponent.PYTHON,))
        assert state.python_state is not None

    def test_default_components(self) -> None:
        """Test default components."""
        random.seed(42)
        state = get_rng_state()
        # Should have at least Python state
        assert state.python_state is not None


class TestSetRNGState:
    """Tests for set_rng_state function."""

    def test_restore_python_state(self) -> None:
        """Test restoring Python RNG state."""
        random.seed(42)
        state = get_rng_state(components=(SeedComponent.PYTHON,))

        # Advance the RNG
        value1 = random.random()

        # Restore state
        result = set_rng_state(state)
        assert result["python"] is True

        # Should reproduce the same value
        value2 = random.random()
        assert value1 == value2

    def test_none_state(self) -> None:
        """Test None state raises error."""
        with pytest.raises(ValueError, match="state cannot be None"):
            set_rng_state(None)  # type: ignore[arg-type]


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_sha256_hash(self) -> None:
        """Test SHA256 hash computation."""
        config = create_reproducibility_config()
        hash_str = compute_config_hash(config)
        assert len(hash_str) == 64  # SHA256 produces 64 hex chars

    def test_md5_hash(self) -> None:
        """Test MD5 hash computation."""
        config = create_reproducibility_config()
        hash_str = compute_config_hash(config, ChecksumType.MD5)
        assert len(hash_str) == 32  # MD5 produces 32 hex chars

    def test_xxhash(self) -> None:
        """Test xxhash computation (falls back to SHA256 if not installed)."""
        config = create_reproducibility_config()
        hash_str = compute_config_hash(config, ChecksumType.XXHASH)
        assert len(hash_str) > 0

    def test_deterministic(self) -> None:
        """Test hash is deterministic."""
        config = create_reproducibility_config()
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2

    def test_different_configs_different_hashes(self) -> None:
        """Test different configs produce different hashes."""
        config1 = create_reproducibility_config(determinism_level="partial")
        config2 = create_reproducibility_config(determinism_level="full")
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 != hash2

    def test_none_config(self) -> None:
        """Test None config raises error."""
        with pytest.raises(ValueError, match="config cannot be None"):
            compute_config_hash(None)  # type: ignore[arg-type]


class TestComputeEnvironmentHash:
    """Tests for compute_environment_hash function."""

    def test_sha256_hash(self) -> None:
        """Test SHA256 hash computation."""
        info = create_environment_info(python_version="3.11.5")
        hash_str = compute_environment_hash(info)
        assert len(hash_str) == 64

    def test_md5_hash(self) -> None:
        """Test MD5 hash computation."""
        info = create_environment_info(python_version="3.11.5")
        hash_str = compute_environment_hash(info, ChecksumType.MD5)
        assert len(hash_str) == 32

    def test_deterministic(self) -> None:
        """Test hash is deterministic."""
        info = create_environment_info(
            python_version="3.11.5",
            torch_version="2.1.0",
        )
        hash1 = compute_environment_hash(info)
        hash2 = compute_environment_hash(info)
        assert hash1 == hash2

    def test_none_info(self) -> None:
        """Test None info raises error."""
        with pytest.raises(ValueError, match="info cannot be None"):
            compute_environment_hash(None)  # type: ignore[arg-type]


class TestVerifyReproducibility:
    """Tests for verify_reproducibility function."""

    def test_valid_config(self) -> None:
        """Test valid config verification."""
        config = create_reproducibility_config()
        config_hash = compute_config_hash(config)
        is_valid, reasons = verify_reproducibility(config, config_hash)
        assert is_valid is True
        assert reasons == []

    def test_invalid_config_hash(self) -> None:
        """Test invalid config hash."""
        config = create_reproducibility_config()
        is_valid, reasons = verify_reproducibility(config, "wrong_hash")
        assert is_valid is False
        assert len(reasons) == 1
        assert "config_hash mismatch" in reasons[0]

    def test_with_environment_hash(self) -> None:
        """Test with environment hash verification."""
        config = create_reproducibility_config()
        config_hash = compute_config_hash(config)
        env_info = create_environment_info(python_version="3.11.5")
        env_hash = compute_environment_hash(env_info)

        is_valid, reasons = verify_reproducibility(
            config, config_hash, env_hash, env_info
        )
        assert is_valid is True
        assert reasons == []

    def test_invalid_environment_hash(self) -> None:
        """Test invalid environment hash."""
        config = create_reproducibility_config()
        config_hash = compute_config_hash(config)
        env_info = create_environment_info(python_version="3.11.5")

        is_valid, reasons = verify_reproducibility(
            config, config_hash, "wrong_env_hash", env_info
        )
        assert is_valid is False
        assert any("env_hash mismatch" in r for r in reasons)

    def test_none_config(self) -> None:
        """Test None config raises error."""
        with pytest.raises(ValueError, match="config cannot be None"):
            verify_reproducibility(None, "hash")  # type: ignore[arg-type]

    def test_empty_expected_hash(self) -> None:
        """Test empty expected hash raises error."""
        config = create_reproducibility_config()
        with pytest.raises(ValueError, match="expected_config_hash cannot be empty"):
            verify_reproducibility(config, "")


class TestFormatReproducibilityStats:
    """Tests for format_reproducibility_stats function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        stats = create_reproducibility_stats(
            seed=42,
            determinism_level=DeterminismLevel.FULL,
            env_hash="abc123",
            config_hash="def456",
        )
        formatted = format_reproducibility_stats(stats)
        assert "Seed: 42" in formatted
        assert "Determinism: full" in formatted
        assert "abc123" in formatted
        assert "def456" in formatted

    def test_none_stats(self) -> None:
        """Test None stats raises error."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_reproducibility_stats(None)  # type: ignore[arg-type]


class TestFormatEnvironmentInfo:
    """Tests for format_environment_info function."""

    def test_full_info(self) -> None:
        """Test formatting full info."""
        info = create_environment_info(
            python_version="3.11.5",
            torch_version="2.1.0",
            cuda_version="12.1",
            transformers_version="4.35.0",
        )
        formatted = format_environment_info(info)
        assert "Python: 3.11.5" in formatted
        assert "PyTorch: 2.1.0" in formatted
        assert "CUDA: 12.1" in formatted
        assert "Transformers: 4.35.0" in formatted

    def test_partial_info(self) -> None:
        """Test formatting partial info."""
        info = create_environment_info(python_version="3.11.5")
        formatted = format_environment_info(info)
        assert "Python: 3.11.5" in formatted
        assert "Not installed" in formatted
        assert "Not available" in formatted

    def test_none_info(self) -> None:
        """Test None info raises error."""
        with pytest.raises(ValueError, match="info cannot be None"):
            format_environment_info(None)  # type: ignore[arg-type]


class TestGetRecommendedReproducibilityConfig:
    """Tests for get_recommended_reproducibility_config function."""

    def test_training_config(self) -> None:
        """Test training configuration."""
        config = get_recommended_reproducibility_config("training")
        assert config.determinism_level == DeterminismLevel.PARTIAL
        assert config.log_environment is True
        assert config.seed_config.seed == 42

    def test_evaluation_config(self) -> None:
        """Test evaluation configuration."""
        config = get_recommended_reproducibility_config("evaluation")
        assert config.determinism_level == DeterminismLevel.FULL
        assert config.seed_config.deterministic_algorithms is True

    def test_debugging_config(self) -> None:
        """Test debugging configuration."""
        config = get_recommended_reproducibility_config("debugging")
        assert config.determinism_level == DeterminismLevel.FULL
        assert config.seed_config.seed == 0
        assert config.seed_config.deterministic_algorithms is True

    def test_no_gpu(self) -> None:
        """Test configuration without GPU."""
        config = get_recommended_reproducibility_config("training", has_gpu=False)
        assert SeedComponent.CUDA not in config.seed_config.components

    def test_invalid_use_case(self) -> None:
        """Test invalid use case."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_reproducibility_config("invalid")
