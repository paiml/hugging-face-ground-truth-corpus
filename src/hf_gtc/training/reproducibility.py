"""Reproducible training utilities for HuggingFace models.

This module provides utilities for ensuring reproducibility in machine learning
training, including seed management, RNG state handling, environment logging,
and configuration hashing.

Examples:
    >>> from hf_gtc.training.reproducibility import (
    ...     create_seed_config,
    ...     create_reproducibility_config,
    ...     SeedComponent,
    ... )
    >>> seed_config = create_seed_config(seed=42)
    >>> seed_config.seed
    42
    >>> config = create_reproducibility_config()
    >>> config.determinism_level
    <DeterminismLevel.PARTIAL: 'partial'>
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class SeedComponent(Enum):
    """Components that can be seeded for reproducibility.

    Attributes:
        PYTHON: Python's built-in random module.
        NUMPY: NumPy's random number generator.
        TORCH: PyTorch's random number generator.
        CUDA: CUDA's random number generator for GPU operations.
        TRANSFORMERS: HuggingFace transformers library.

    Examples:
        >>> SeedComponent.PYTHON.value
        'python'
        >>> SeedComponent.NUMPY.value
        'numpy'
        >>> SeedComponent.TORCH.value
        'torch'
        >>> SeedComponent.CUDA.value
        'cuda'
        >>> SeedComponent.TRANSFORMERS.value
        'transformers'
    """

    PYTHON = "python"
    NUMPY = "numpy"
    TORCH = "torch"
    CUDA = "cuda"
    TRANSFORMERS = "transformers"


class DeterminismLevel(Enum):
    """Levels of determinism for training.

    Attributes:
        NONE: No determinism guarantees.
        PARTIAL: Some deterministic operations (seeds set, but CUDA non-deterministic).
        FULL: Full determinism (may impact performance).

    Examples:
        >>> DeterminismLevel.NONE.value
        'none'
        >>> DeterminismLevel.PARTIAL.value
        'partial'
        >>> DeterminismLevel.FULL.value
        'full'
    """

    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class ChecksumType(Enum):
    """Types of checksums for configuration hashing.

    Attributes:
        MD5: MD5 hash (fast, not cryptographically secure).
        SHA256: SHA-256 hash (secure, recommended).
        XXHASH: xxHash (very fast, good for non-security uses).

    Examples:
        >>> ChecksumType.MD5.value
        'md5'
        >>> ChecksumType.SHA256.value
        'sha256'
        >>> ChecksumType.XXHASH.value
        'xxhash'
    """

    MD5 = "md5"
    SHA256 = "sha256"
    XXHASH = "xxhash"


VALID_SEED_COMPONENTS = frozenset(c.value for c in SeedComponent)
VALID_DETERMINISM_LEVELS = frozenset(d.value for d in DeterminismLevel)
VALID_CHECKSUM_TYPES = frozenset(c.value for c in ChecksumType)


@dataclass(frozen=True, slots=True)
class SeedConfig:
    """Configuration for random seed management.

    Attributes:
        seed: The random seed value.
        components: Tuple of components to seed.
        deterministic_algorithms: Whether to use deterministic algorithms.

    Examples:
        >>> config = SeedConfig(
        ...     seed=42,
        ...     components=(SeedComponent.PYTHON, SeedComponent.NUMPY),
        ...     deterministic_algorithms=False,
        ... )
        >>> config.seed
        42
        >>> SeedComponent.PYTHON in config.components
        True

        >>> config2 = SeedConfig(
        ...     seed=123,
        ...     components=(SeedComponent.TORCH, SeedComponent.CUDA),
        ...     deterministic_algorithms=True,
        ... )
        >>> config2.deterministic_algorithms
        True
    """

    seed: int
    components: tuple[SeedComponent, ...]
    deterministic_algorithms: bool


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Main configuration for reproducibility settings.

    Attributes:
        seed_config: Configuration for seeding.
        determinism_level: Level of determinism to enforce.
        log_environment: Whether to log environment information.
        save_rng_state: Whether to save RNG states to checkpoints.

    Examples:
        >>> seed_cfg = SeedConfig(42, (SeedComponent.PYTHON,), False)
        >>> config = ReproducibilityConfig(
        ...     seed_config=seed_cfg,
        ...     determinism_level=DeterminismLevel.PARTIAL,
        ...     log_environment=True,
        ...     save_rng_state=True,
        ... )
        >>> config.log_environment
        True
        >>> config.determinism_level
        <DeterminismLevel.PARTIAL: 'partial'>
    """

    seed_config: SeedConfig
    determinism_level: DeterminismLevel
    log_environment: bool
    save_rng_state: bool


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Information about the runtime environment.

    Attributes:
        python_version: Python version string.
        torch_version: PyTorch version (or None if not installed).
        cuda_version: CUDA version (or None if not available).
        transformers_version: Transformers version (or None if not installed).

    Examples:
        >>> info = EnvironmentInfo(
        ...     python_version="3.11.5",
        ...     torch_version="2.1.0",
        ...     cuda_version="12.1",
        ...     transformers_version="4.35.0",
        ... )
        >>> info.python_version
        '3.11.5'
        >>> info.torch_version
        '2.1.0'

        >>> info2 = EnvironmentInfo(
        ...     python_version="3.10.0",
        ...     torch_version=None,
        ...     cuda_version=None,
        ...     transformers_version=None,
        ... )
        >>> info2.torch_version is None
        True
    """

    python_version: str
    torch_version: str | None
    cuda_version: str | None
    transformers_version: str | None


@dataclass(frozen=True, slots=True)
class RNGState:
    """Container for random number generator states.

    Attributes:
        python_state: Python random state (serialized).
        numpy_state: NumPy random state (serialized).
        torch_state: PyTorch CPU RNG state (serialized).
        cuda_state: PyTorch CUDA RNG state (serialized, per device).

    Examples:
        >>> state = RNGState(
        ...     python_state=b"python_state_data",
        ...     numpy_state=b"numpy_state_data",
        ...     torch_state=b"torch_state_data",
        ...     cuda_state=None,
        ... )
        >>> state.python_state
        b'python_state_data'
        >>> state.cuda_state is None
        True
    """

    python_state: bytes | None
    numpy_state: bytes | None
    torch_state: bytes | None
    cuda_state: bytes | None


@dataclass(frozen=True, slots=True)
class ReproducibilityStats:
    """Statistics and hashes for reproducibility verification.

    Attributes:
        seed: The seed used for training.
        determinism_level: The determinism level used.
        env_hash: Hash of the environment information.
        config_hash: Hash of the configuration.

    Examples:
        >>> stats = ReproducibilityStats(
        ...     seed=42,
        ...     determinism_level=DeterminismLevel.FULL,
        ...     env_hash="abc123",
        ...     config_hash="def456",
        ... )
        >>> stats.seed
        42
        >>> stats.env_hash
        'abc123'
    """

    seed: int
    determinism_level: DeterminismLevel
    env_hash: str
    config_hash: str


def validate_seed_config(config: SeedConfig) -> None:
    """Validate seed configuration.

    Args:
        config: Seed configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If seed is negative.
        ValueError: If components is empty.

    Examples:
        >>> config = SeedConfig(42, (SeedComponent.PYTHON,), False)
        >>> validate_seed_config(config)

        >>> validate_seed_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SeedConfig(-1, (SeedComponent.PYTHON,), False)
        >>> validate_seed_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: seed must be non-negative
    """
    validate_not_none(config, "config")

    if config.seed < 0:
        msg = f"seed must be non-negative, got {config.seed}"
        raise ValueError(msg)

    if not config.components:
        msg = "components cannot be empty"
        raise ValueError(msg)


def validate_reproducibility_config(config: ReproducibilityConfig) -> None:
    """Validate reproducibility configuration.

    Args:
        config: Reproducibility configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If seed_config is invalid.

    Examples:
        >>> seed_cfg = SeedConfig(42, (SeedComponent.PYTHON,), False)
        >>> config = ReproducibilityConfig(
        ...     seed_cfg, DeterminismLevel.PARTIAL, True, True
        ... )
        >>> validate_reproducibility_config(config)

        >>> validate_reproducibility_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_seed_config(config.seed_config)


def validate_environment_info(info: EnvironmentInfo) -> None:
    """Validate environment information.

    Args:
        info: Environment information to validate.

    Raises:
        ValueError: If info is None.
        ValueError: If python_version is empty.

    Examples:
        >>> info = EnvironmentInfo("3.11.5", "2.1.0", "12.1", "4.35.0")
        >>> validate_environment_info(info)

        >>> validate_environment_info(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: info cannot be None

        >>> bad = EnvironmentInfo("", None, None, None)
        >>> validate_environment_info(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: python_version cannot be empty
    """
    if info is None:
        msg = "info cannot be None"
        raise ValueError(msg)

    if not info.python_version:
        msg = "python_version cannot be empty"
        raise ValueError(msg)


def validate_rng_state(state: RNGState) -> None:
    """Validate RNG state container.

    Args:
        state: RNG state to validate.

    Raises:
        ValueError: If state is None.
        ValueError: If all states are None.

    Examples:
        >>> state = RNGState(b"data", None, None, None)
        >>> validate_rng_state(state)

        >>> validate_rng_state(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: state cannot be None

        >>> empty = RNGState(None, None, None, None)
        >>> validate_rng_state(empty)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: at least one RNG state must be provided
    """
    if state is None:
        msg = "state cannot be None"
        raise ValueError(msg)

    if all(
        s is None
        for s in [
            state.python_state,
            state.numpy_state,
            state.torch_state,
            state.cuda_state,
        ]
    ):
        msg = "at least one RNG state must be provided"
        raise ValueError(msg)


def validate_reproducibility_stats(stats: ReproducibilityStats) -> None:
    """Validate reproducibility statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If seed is negative.
        ValueError: If env_hash is empty.
        ValueError: If config_hash is empty.

    Examples:
        >>> stats = ReproducibilityStats(
        ...     42, DeterminismLevel.PARTIAL, "abc", "def"
        ... )
        >>> validate_reproducibility_stats(stats)

        >>> validate_reproducibility_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = ReproducibilityStats(-1, DeterminismLevel.PARTIAL, "abc", "def")
        >>> validate_reproducibility_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: seed must be non-negative
    """
    validate_not_none(stats, "stats")

    if stats.seed < 0:
        msg = f"seed must be non-negative, got {stats.seed}"
        raise ValueError(msg)

    if not stats.env_hash:
        msg = "env_hash cannot be empty"
        raise ValueError(msg)

    if not stats.config_hash:
        msg = "config_hash cannot be empty"
        raise ValueError(msg)


def create_seed_config(
    seed: int = 42,
    components: tuple[SeedComponent, ...] | None = None,
    deterministic_algorithms: bool = False,
) -> SeedConfig:
    """Create a seed configuration with validation.

    Args:
        seed: Random seed value. Defaults to 42.
        components: Components to seed. Defaults to all components.
        deterministic_algorithms: Use deterministic algorithms. Defaults to False.

    Returns:
        Validated SeedConfig.

    Raises:
        ValueError: If seed is negative.
        ValueError: If components is empty.

    Examples:
        >>> config = create_seed_config()
        >>> config.seed
        42
        >>> len(config.components) == len(SeedComponent)
        True

        >>> config2 = create_seed_config(seed=123, deterministic_algorithms=True)
        >>> config2.seed
        123
        >>> config2.deterministic_algorithms
        True

        >>> config3 = create_seed_config(components=(SeedComponent.PYTHON,))
        >>> len(config3.components)
        1

        >>> create_seed_config(seed=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: seed must be non-negative
    """
    if components is None:
        components = tuple(SeedComponent)

    config = SeedConfig(
        seed=seed,
        components=components,
        deterministic_algorithms=deterministic_algorithms,
    )
    validate_seed_config(config)
    return config


def create_reproducibility_config(
    seed_config: SeedConfig | None = None,
    determinism_level: str | DeterminismLevel = DeterminismLevel.PARTIAL,
    log_environment: bool = True,
    save_rng_state: bool = True,
) -> ReproducibilityConfig:
    """Create a reproducibility configuration with validation.

    Args:
        seed_config: Seed configuration. Defaults to create_seed_config().
        determinism_level: Level of determinism. Defaults to "partial".
        log_environment: Log environment info. Defaults to True.
        save_rng_state: Save RNG states to checkpoints. Defaults to True.

    Returns:
        Validated ReproducibilityConfig.

    Raises:
        ValueError: If determinism_level is invalid.

    Examples:
        >>> config = create_reproducibility_config()
        >>> config.determinism_level
        <DeterminismLevel.PARTIAL: 'partial'>
        >>> config.log_environment
        True

        >>> config2 = create_reproducibility_config(determinism_level="full")
        >>> config2.determinism_level
        <DeterminismLevel.FULL: 'full'>

        >>> seed_cfg = create_seed_config(seed=123)
        >>> config3 = create_reproducibility_config(seed_config=seed_cfg)
        >>> config3.seed_config.seed
        123

        >>> create_reproducibility_config(determinism_level="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: determinism_level must be one of
    """
    if seed_config is None:
        seed_config = create_seed_config()

    if isinstance(determinism_level, str):
        determinism_level = get_determinism_level(determinism_level)

    config = ReproducibilityConfig(
        seed_config=seed_config,
        determinism_level=determinism_level,
        log_environment=log_environment,
        save_rng_state=save_rng_state,
    )
    validate_reproducibility_config(config)
    return config


def create_environment_info(
    python_version: str | None = None,
    torch_version: str | None = None,
    cuda_version: str | None = None,
    transformers_version: str | None = None,
) -> EnvironmentInfo:
    """Create environment information, auto-detecting if not provided.

    Args:
        python_version: Python version. Defaults to current version.
        torch_version: PyTorch version. Defaults to None.
        cuda_version: CUDA version. Defaults to None.
        transformers_version: Transformers version. Defaults to None.

    Returns:
        Validated EnvironmentInfo.

    Raises:
        ValueError: If python_version is empty after resolution.

    Examples:
        >>> info = create_environment_info()
        >>> len(info.python_version) > 0
        True

        >>> info2 = create_environment_info(
        ...     python_version="3.11.5",
        ...     torch_version="2.1.0",
        ... )
        >>> info2.python_version
        '3.11.5'
        >>> info2.torch_version
        '2.1.0'
    """
    if python_version is None:
        python_version = platform.python_version()

    info = EnvironmentInfo(
        python_version=python_version,
        torch_version=torch_version,
        cuda_version=cuda_version,
        transformers_version=transformers_version,
    )
    validate_environment_info(info)
    return info


def create_rng_state(
    python_state: bytes | None = None,
    numpy_state: bytes | None = None,
    torch_state: bytes | None = None,
    cuda_state: bytes | None = None,
) -> RNGState:
    """Create an RNG state container with validation.

    Args:
        python_state: Python random state.
        numpy_state: NumPy random state.
        torch_state: PyTorch CPU RNG state.
        cuda_state: PyTorch CUDA RNG state.

    Returns:
        Validated RNGState.

    Raises:
        ValueError: If all states are None.

    Examples:
        >>> state = create_rng_state(python_state=b"test")
        >>> state.python_state
        b'test'

        >>> state2 = create_rng_state(numpy_state=b"np", torch_state=b"torch")
        >>> state2.numpy_state
        b'np'

        >>> create_rng_state()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: at least one RNG state must be provided
    """
    state = RNGState(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_state=torch_state,
        cuda_state=cuda_state,
    )
    validate_rng_state(state)
    return state


def create_reproducibility_stats(
    seed: int = 42,
    determinism_level: DeterminismLevel = DeterminismLevel.PARTIAL,
    env_hash: str = "",
    config_hash: str = "",
) -> ReproducibilityStats:
    """Create reproducibility statistics with validation.

    Args:
        seed: The seed used.
        determinism_level: The determinism level used.
        env_hash: Hash of environment info.
        config_hash: Hash of configuration.

    Returns:
        Validated ReproducibilityStats.

    Raises:
        ValueError: If seed is negative.
        ValueError: If env_hash is empty.
        ValueError: If config_hash is empty.

    Examples:
        >>> stats = create_reproducibility_stats(
        ...     seed=42,
        ...     determinism_level=DeterminismLevel.FULL,
        ...     env_hash="abc123",
        ...     config_hash="def456",
        ... )
        >>> stats.seed
        42
        >>> stats.env_hash
        'abc123'

        >>> create_reproducibility_stats(env_hash="", config_hash="x")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: env_hash cannot be empty
    """
    stats = ReproducibilityStats(
        seed=seed,
        determinism_level=determinism_level,
        env_hash=env_hash,
        config_hash=config_hash,
    )
    validate_reproducibility_stats(stats)
    return stats


def list_seed_components() -> list[str]:
    """List all available seed components.

    Returns:
        Sorted list of seed component names.

    Examples:
        >>> components = list_seed_components()
        >>> "python" in components
        True
        >>> "torch" in components
        True
        >>> components == sorted(components)
        True
    """
    return sorted(VALID_SEED_COMPONENTS)


def list_determinism_levels() -> list[str]:
    """List all available determinism levels.

    Returns:
        Sorted list of determinism level names.

    Examples:
        >>> levels = list_determinism_levels()
        >>> "none" in levels
        True
        >>> "full" in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_DETERMINISM_LEVELS)


def list_checksum_types() -> list[str]:
    """List all available checksum types.

    Returns:
        Sorted list of checksum type names.

    Examples:
        >>> types = list_checksum_types()
        >>> "md5" in types
        True
        >>> "sha256" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_CHECKSUM_TYPES)


def get_seed_component(name: str) -> SeedComponent:
    """Get seed component enum from string name.

    Args:
        name: Name of the seed component.

    Returns:
        Corresponding SeedComponent enum.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_seed_component("python")
        <SeedComponent.PYTHON: 'python'>
        >>> get_seed_component("torch")
        <SeedComponent.TORCH: 'torch'>

        >>> get_seed_component("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: seed_component must be one of
    """
    if name not in VALID_SEED_COMPONENTS:
        msg = f"seed_component must be one of {VALID_SEED_COMPONENTS}, got '{name}'"
        raise ValueError(msg)
    return SeedComponent(name)


def get_determinism_level(name: str) -> DeterminismLevel:
    """Get determinism level enum from string name.

    Args:
        name: Name of the determinism level.

    Returns:
        Corresponding DeterminismLevel enum.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_determinism_level("none")
        <DeterminismLevel.NONE: 'none'>
        >>> get_determinism_level("full")
        <DeterminismLevel.FULL: 'full'>

        >>> get_determinism_level("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: determinism_level must be one of
    """
    if name not in VALID_DETERMINISM_LEVELS:
        msg = (
            f"determinism_level must be one of {VALID_DETERMINISM_LEVELS}, got '{name}'"
        )
        raise ValueError(msg)
    return DeterminismLevel(name)


def get_checksum_type(name: str) -> ChecksumType:
    """Get checksum type enum from string name.

    Args:
        name: Name of the checksum type.

    Returns:
        Corresponding ChecksumType enum.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_checksum_type("md5")
        <ChecksumType.MD5: 'md5'>
        >>> get_checksum_type("sha256")
        <ChecksumType.SHA256: 'sha256'>

        >>> get_checksum_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: checksum_type must be one of
    """
    if name not in VALID_CHECKSUM_TYPES:
        msg = f"checksum_type must be one of {VALID_CHECKSUM_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ChecksumType(name)


def _seed_python(seed: int, config: SeedConfig, results: dict[str, bool]) -> None:
    """Seed the Python random module."""
    import random

    random.seed(seed)
    results["python"] = True


def _seed_numpy(seed: int, config: SeedConfig, results: dict[str, bool]) -> None:
    """Seed the NumPy random module."""
    try:
        import numpy as np

        np.random.seed(seed)
        results["numpy"] = True
    except ImportError:
        results["numpy"] = False


def _seed_torch(seed: int, config: SeedConfig, results: dict[str, bool]) -> None:
    """Seed PyTorch."""
    try:
        import torch

        torch.manual_seed(seed)
        if config.deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        results["torch"] = True
    except ImportError:
        results["torch"] = False


def _seed_cuda(seed: int, config: SeedConfig, results: dict[str, bool]) -> None:
    """Seed CUDA RNG."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if config.deterministic_algorithms:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            results["cuda"] = True
        else:
            results["cuda"] = False
    except ImportError:
        results["cuda"] = False


def _seed_transformers(seed: int, config: SeedConfig, results: dict[str, bool]) -> None:
    """Seed Hugging Face transformers."""
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
        results["transformers"] = True
    except ImportError:
        results["transformers"] = False


def set_all_seeds(config: SeedConfig) -> dict[str, bool]:
    """Set random seeds for all configured components.

    This function sets seeds for Python, NumPy, PyTorch, and CUDA
    based on the provided configuration.

    Args:
        config: Seed configuration specifying which components to seed.

    Returns:
        Dictionary mapping component names to success status.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_seed_config(seed=42, components=(SeedComponent.PYTHON,))
        >>> result = set_all_seeds(config)
        >>> result["python"]
        True

        >>> set_all_seeds(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_seed_config(config)

    results: dict[str, bool] = {}
    seed = config.seed

    seeders = {
        SeedComponent.PYTHON: _seed_python,
        SeedComponent.NUMPY: _seed_numpy,
        SeedComponent.TORCH: _seed_torch,
        SeedComponent.CUDA: _seed_cuda,
        SeedComponent.TRANSFORMERS: _seed_transformers,
    }

    for component in config.components:
        seeder = seeders.get(component)
        if seeder is not None:
            seeder(seed, config, results)

    return results


def get_rng_state(components: tuple[SeedComponent, ...] | None = None) -> RNGState:
    """Get current RNG states for specified components.

    Args:
        components: Components to get state for. Defaults to all.

    Returns:
        RNGState containing the current states.

    Examples:
        >>> import random
        >>> random.seed(42)
        >>> state = get_rng_state(components=(SeedComponent.PYTHON,))
        >>> state.python_state is not None
        True
    """
    if components is None:
        components = tuple(SeedComponent)

    python_state: bytes | None = None
    numpy_state: bytes | None = None
    torch_state: bytes | None = None
    cuda_state: bytes | None = None

    for component in components:
        if component == SeedComponent.PYTHON:
            import pickle
            import random

            python_state = pickle.dumps(random.getstate())

        elif component == SeedComponent.NUMPY:
            try:
                import pickle

                import numpy as np

                numpy_state = pickle.dumps(np.random.get_state())
            except ImportError:
                pass

        elif component == SeedComponent.TORCH:
            try:
                import pickle

                import torch

                torch_state = pickle.dumps(torch.get_rng_state())
            except ImportError:
                pass

        elif component == SeedComponent.CUDA:
            try:
                import pickle

                import torch

                if torch.cuda.is_available():
                    cuda_state = pickle.dumps(torch.cuda.get_rng_state_all())
            except ImportError:
                pass

    # Ensure at least one state is set
    if all(s is None for s in [python_state, numpy_state, torch_state, cuda_state]):
        # Default to getting Python state
        import pickle
        import random

        python_state = pickle.dumps(random.getstate())

    return RNGState(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_state=torch_state,
        cuda_state=cuda_state,
    )


def _restore_python_state(saved: bytes, results: dict[str, bool], key: str) -> None:
    """Restore Python RNG state."""
    import pickle
    import random

    random.setstate(pickle.loads(saved))  # nosec B301 - trusted internal state
    results[key] = True


def _restore_numpy_state(saved: bytes, results: dict[str, bool], key: str) -> None:
    """Restore NumPy RNG state."""
    import pickle

    try:
        import numpy as np

        np.random.set_state(pickle.loads(saved))  # nosec B301
        results[key] = True
    except ImportError:
        results[key] = False


def _restore_torch_state(saved: bytes, results: dict[str, bool], key: str) -> None:
    """Restore PyTorch RNG state."""
    import pickle

    try:
        import torch

        torch.set_rng_state(pickle.loads(saved))  # nosec B301
        results[key] = True
    except ImportError:
        results[key] = False


def _restore_cuda_state(saved: bytes, results: dict[str, bool], key: str) -> None:
    """Restore CUDA RNG state."""
    import pickle

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(pickle.loads(saved))  # nosec B301
            results[key] = True
        else:
            results[key] = False
    except ImportError:
        results[key] = False


def set_rng_state(state: RNGState) -> dict[str, bool]:
    """Restore RNG states from a saved state container.

    Args:
        state: RNGState containing saved states.

    Returns:
        Dictionary mapping component names to success status.

    Raises:
        ValueError: If state is None.

    Examples:
        >>> import random
        >>> random.seed(42)
        >>> state = get_rng_state(components=(SeedComponent.PYTHON,))
        >>> random.random()  # Advance state
        0.6394267984578837
        >>> result = set_rng_state(state)
        >>> result["python"]
        True
        >>> random.random()  # Should reproduce
        0.6394267984578837
    """
    if state is None:
        msg = "state cannot be None"
        raise ValueError(msg)

    results: dict[str, bool] = {}

    restorers: tuple[tuple[str, str, object], ...] = (
        ("python_state", "python", _restore_python_state),
        ("numpy_state", "numpy", _restore_numpy_state),
        ("torch_state", "torch", _restore_torch_state),
        ("cuda_state", "cuda", _restore_cuda_state),
    )

    for attr_name, key, restorer in restorers:
        saved = getattr(state, attr_name)
        if saved is not None:
            restorer(saved, results, key)

    return results


def compute_config_hash(
    config: ReproducibilityConfig,
    checksum_type: ChecksumType = ChecksumType.SHA256,
) -> str:
    """Compute a hash of the reproducibility configuration.

    Args:
        config: Configuration to hash.
        checksum_type: Type of checksum to use.

    Returns:
        Hexadecimal hash string.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_reproducibility_config()
        >>> hash_str = compute_config_hash(config)
        >>> len(hash_str) == 64  # SHA256 produces 64 hex chars
        True

        >>> hash2 = compute_config_hash(config, ChecksumType.MD5)
        >>> len(hash2) == 32  # MD5 produces 32 hex chars
        True

        >>> compute_config_hash(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    # Create a canonical string representation
    components_str = ",".join(sorted(c.value for c in config.seed_config.components))
    parts = [
        f"seed={config.seed_config.seed}",
        f"deterministic_algorithms={config.seed_config.deterministic_algorithms}",
        f"components={components_str}",
        f"determinism_level={config.determinism_level.value}",
        f"log_environment={config.log_environment}",
        f"save_rng_state={config.save_rng_state}",
    ]
    data = "|".join(parts).encode("utf-8")

    if checksum_type == ChecksumType.MD5:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    elif checksum_type == ChecksumType.SHA256:
        return hashlib.sha256(data).hexdigest()
    else:  # XXHASH
        # Fall back to SHA256 if xxhash is not available
        try:
            import xxhash

            return xxhash.xxh64(data).hexdigest()
        except ImportError:
            return hashlib.sha256(data).hexdigest()


def compute_environment_hash(
    info: EnvironmentInfo,
    checksum_type: ChecksumType = ChecksumType.SHA256,
) -> str:
    """Compute a hash of the environment information.

    Args:
        info: Environment information to hash.
        checksum_type: Type of checksum to use.

    Returns:
        Hexadecimal hash string.

    Raises:
        ValueError: If info is None.

    Examples:
        >>> info = create_environment_info(
        ...     python_version="3.11.5",
        ...     torch_version="2.1.0",
        ... )
        >>> hash_str = compute_environment_hash(info)
        >>> len(hash_str) == 64
        True

        >>> compute_environment_hash(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: info cannot be None
    """
    if info is None:
        msg = "info cannot be None"
        raise ValueError(msg)

    parts = [
        f"python={info.python_version}",
        f"torch={info.torch_version or 'None'}",
        f"cuda={info.cuda_version or 'None'}",
        f"transformers={info.transformers_version or 'None'}",
    ]
    data = "|".join(parts).encode("utf-8")

    if checksum_type == ChecksumType.MD5:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    elif checksum_type == ChecksumType.SHA256:
        return hashlib.sha256(data).hexdigest()
    else:  # XXHASH
        try:
            import xxhash

            return xxhash.xxh64(data).hexdigest()
        except ImportError:
            return hashlib.sha256(data).hexdigest()


def verify_reproducibility(
    config: ReproducibilityConfig,
    expected_config_hash: str,
    expected_env_hash: str | None = None,
    current_env: EnvironmentInfo | None = None,
) -> tuple[bool, list[str]]:
    """Verify that current configuration matches expected hashes.

    Args:
        config: Current reproducibility configuration.
        expected_config_hash: Expected configuration hash.
        expected_env_hash: Expected environment hash (optional).
        current_env: Current environment info (optional).

    Returns:
        Tuple of (is_valid, list of mismatch reasons).

    Raises:
        ValueError: If config is None.
        ValueError: If expected_config_hash is empty.

    Examples:
        >>> config = create_reproducibility_config()
        >>> config_hash = compute_config_hash(config)
        >>> is_valid, reasons = verify_reproducibility(config, config_hash)
        >>> is_valid
        True
        >>> reasons
        []

        >>> is_valid, reasons = verify_reproducibility(config, "wrong_hash")
        >>> is_valid
        False
        >>> "config_hash mismatch" in reasons[0]
        True

        >>> verify_reproducibility(None, "hash")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if not expected_config_hash:
        msg = "expected_config_hash cannot be empty"
        raise ValueError(msg)

    mismatches: list[str] = []

    # Verify config hash
    current_config_hash = compute_config_hash(config)
    if current_config_hash != expected_config_hash:
        mismatches.append(
            f"config_hash mismatch: expected {expected_config_hash}, "
            f"got {current_config_hash}"
        )

    # Verify environment hash if provided
    if expected_env_hash is not None and current_env is not None:
        current_env_hash = compute_environment_hash(current_env)
        if current_env_hash != expected_env_hash:
            mismatches.append(
                f"env_hash mismatch: expected {expected_env_hash}, "
                f"got {current_env_hash}"
            )

    return len(mismatches) == 0, mismatches


def format_reproducibility_stats(stats: ReproducibilityStats) -> str:
    """Format reproducibility statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = create_reproducibility_stats(
        ...     seed=42,
        ...     determinism_level=DeterminismLevel.FULL,
        ...     env_hash="abc123def456",
        ...     config_hash="789xyz000111",
        ... )
        >>> formatted = format_reproducibility_stats(stats)
        >>> "Seed: 42" in formatted
        True
        >>> "Determinism: full" in formatted
        True
        >>> "abc123def456" in formatted
        True

        >>> format_reproducibility_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    return (
        f"Reproducibility Stats:\n"
        f"  Seed: {stats.seed}\n"
        f"  Determinism: {stats.determinism_level.value}\n"
        f"  Environment Hash: {stats.env_hash}\n"
        f"  Config Hash: {stats.config_hash}"
    )


def format_environment_info(info: EnvironmentInfo) -> str:
    """Format environment information as a human-readable string.

    Args:
        info: Environment information to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If info is None.

    Examples:
        >>> info = create_environment_info(
        ...     python_version="3.11.5",
        ...     torch_version="2.1.0",
        ...     cuda_version="12.1",
        ...     transformers_version="4.35.0",
        ... )
        >>> formatted = format_environment_info(info)
        >>> "Python: 3.11.5" in formatted
        True
        >>> "PyTorch: 2.1.0" in formatted
        True

        >>> format_environment_info(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: info cannot be None
    """
    if info is None:
        msg = "info cannot be None"
        raise ValueError(msg)

    torch_str = info.torch_version or "Not installed"
    cuda_str = info.cuda_version or "Not available"
    transformers_str = info.transformers_version or "Not installed"

    return (
        f"Environment Information:\n"
        f"  Python: {info.python_version}\n"
        f"  PyTorch: {torch_str}\n"
        f"  CUDA: {cuda_str}\n"
        f"  Transformers: {transformers_str}"
    )


def get_recommended_reproducibility_config(
    use_case: str = "training",
    has_gpu: bool = True,
) -> ReproducibilityConfig:
    """Get recommended reproducibility configuration for a use case.

    Args:
        use_case: Type of use case ("training", "evaluation", "debugging").
        has_gpu: Whether GPU is available. Defaults to True.

    Returns:
        Recommended ReproducibilityConfig.

    Raises:
        ValueError: If use_case is invalid.

    Examples:
        >>> config = get_recommended_reproducibility_config()
        >>> config.determinism_level
        <DeterminismLevel.PARTIAL: 'partial'>

        >>> config2 = get_recommended_reproducibility_config("debugging")
        >>> config2.determinism_level
        <DeterminismLevel.FULL: 'full'>

        >>> config3 = get_recommended_reproducibility_config("evaluation")
        >>> config3.save_rng_state
        True

        >>> get_recommended_reproducibility_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case must be one of
    """
    valid_use_cases = frozenset({"training", "evaluation", "debugging"})

    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    # Build components list based on GPU availability
    if has_gpu:
        components = tuple(SeedComponent)
    else:
        components = (
            SeedComponent.PYTHON,
            SeedComponent.NUMPY,
            SeedComponent.TORCH,
            SeedComponent.TRANSFORMERS,
        )

    if use_case == "training":
        # Standard training: partial determinism for good balance
        seed_config = create_seed_config(
            seed=42,
            components=components,
            deterministic_algorithms=False,
        )
        return create_reproducibility_config(
            seed_config=seed_config,
            determinism_level=DeterminismLevel.PARTIAL,
            log_environment=True,
            save_rng_state=True,
        )

    elif use_case == "evaluation":
        # Evaluation: full determinism for consistent results
        seed_config = create_seed_config(
            seed=42,
            components=components,
            deterministic_algorithms=True,
        )
        return create_reproducibility_config(
            seed_config=seed_config,
            determinism_level=DeterminismLevel.FULL,
            log_environment=True,
            save_rng_state=True,
        )

    else:  # debugging
        # Debugging: full determinism with detailed logging
        seed_config = create_seed_config(
            seed=0,  # Simple seed for debugging
            components=components,
            deterministic_algorithms=True,
        )
        return create_reproducibility_config(
            seed_config=seed_config,
            determinism_level=DeterminismLevel.FULL,
            log_environment=True,
            save_rng_state=True,
        )
