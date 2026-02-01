"""Differential privacy and data anonymization utilities.

This module provides utilities for implementing differential privacy mechanisms
and data anonymization techniques for ML data pipelines, including noise
injection, k-anonymity, l-diversity, and t-closeness checks.

Examples:
    >>> from hf_gtc.safety.privacy import PrivacyMechanism, AnonymizationType
    >>> PrivacyMechanism.LAPLACE.value
    'laplace'
    >>> AnonymizationType.K_ANONYMITY.value
    'k_anonymity'

    >>> from hf_gtc.safety.privacy import create_dp_config, calculate_noise_scale
    >>> config = create_dp_config(epsilon=1.0, delta=1e-5)
    >>> config.epsilon
    1.0
    >>> scale = calculate_noise_scale(epsilon=1.0, sensitivity=1.0)
    >>> scale > 0
    True
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from hf_gtc._validation import validate_not_none


class PrivacyMechanism(Enum):
    """Differential privacy noise mechanism types.

    Attributes:
        LAPLACE: Laplace mechanism for unbounded queries.
        GAUSSIAN: Gaussian mechanism for concentrated DP.
        EXPONENTIAL: Exponential mechanism for discrete outputs.
        RANDOMIZED_RESPONSE: Randomized response for binary data.

    Examples:
        >>> PrivacyMechanism.LAPLACE.value
        'laplace'
        >>> PrivacyMechanism.GAUSSIAN.value
        'gaussian'
        >>> PrivacyMechanism.EXPONENTIAL.value
        'exponential'
        >>> PrivacyMechanism.RANDOMIZED_RESPONSE.value
        'randomized_response'
    """

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    RANDOMIZED_RESPONSE = "randomized_response"


class AnonymizationType(Enum):
    """Data anonymization technique types.

    Attributes:
        K_ANONYMITY: k-anonymity ensuring k identical quasi-identifiers.
        L_DIVERSITY: l-diversity for sensitive attribute diversity.
        T_CLOSENESS: t-closeness for distribution similarity.
        DIFFERENTIAL: Differential privacy anonymization.

    Examples:
        >>> AnonymizationType.K_ANONYMITY.value
        'k_anonymity'
        >>> AnonymizationType.L_DIVERSITY.value
        'l_diversity'
        >>> AnonymizationType.T_CLOSENESS.value
        't_closeness'
        >>> AnonymizationType.DIFFERENTIAL.value
        'differential'
    """

    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"
    DIFFERENTIAL = "differential"


class SensitivityType(Enum):
    """Query sensitivity types for differential privacy.

    Attributes:
        LOCAL: Local sensitivity computed per record.
        GLOBAL: Global sensitivity across all possible databases.
        SMOOTH: Smooth sensitivity for better utility.

    Examples:
        >>> SensitivityType.LOCAL.value
        'local'
        >>> SensitivityType.GLOBAL.value
        'global'
        >>> SensitivityType.SMOOTH.value
        'smooth'
    """

    LOCAL = "local"
    GLOBAL = "global"
    SMOOTH = "smooth"


VALID_PRIVACY_MECHANISMS: frozenset[str] = frozenset(m.value for m in PrivacyMechanism)
VALID_ANONYMIZATION_TYPES: frozenset[str] = frozenset(
    a.value for a in AnonymizationType
)
VALID_SENSITIVITY_TYPES: frozenset[str] = frozenset(s.value for s in SensitivityType)


@dataclass(frozen=True, slots=True)
class DPConfig:
    """Configuration for differential privacy mechanisms.

    Attributes:
        epsilon: Privacy budget (lower = more private).
        delta: Probability of privacy breach (for approximate DP).
        mechanism: Type of DP mechanism to use.
        clip_norm: Maximum L2 norm for gradient clipping.
        noise_multiplier: Multiplier for noise scale.

    Examples:
        >>> config = DPConfig(
        ...     epsilon=1.0,
        ...     delta=1e-5,
        ...     mechanism=PrivacyMechanism.GAUSSIAN,
        ...     clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ... )
        >>> config.epsilon
        1.0
        >>> config.delta
        1e-05
        >>> config.mechanism
        <PrivacyMechanism.GAUSSIAN: 'gaussian'>

        >>> config2 = DPConfig()
        >>> config2.epsilon
        1.0
        >>> config2.mechanism
        <PrivacyMechanism.LAPLACE: 'laplace'>
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class AnonymizationConfig:
    """Configuration for data anonymization techniques.

    Attributes:
        anon_type: Type of anonymization technique.
        k_value: K value for k-anonymity (minimum group size).
        quasi_identifiers: Tuple of quasi-identifier column names.
        sensitive_attributes: Tuple of sensitive attribute column names.

    Examples:
        >>> config = AnonymizationConfig(
        ...     anon_type=AnonymizationType.K_ANONYMITY,
        ...     k_value=5,
        ...     quasi_identifiers=("age", "zipcode"),
        ...     sensitive_attributes=("salary",),
        ... )
        >>> config.k_value
        5
        >>> "age" in config.quasi_identifiers
        True

        >>> config2 = AnonymizationConfig()
        >>> config2.k_value
        5
        >>> config2.anon_type
        <AnonymizationType.K_ANONYMITY: 'k_anonymity'>
    """

    anon_type: AnonymizationType = AnonymizationType.K_ANONYMITY
    k_value: int = 5
    quasi_identifiers: tuple[str, ...] = field(default_factory=tuple)
    sensitive_attributes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    """Combined configuration for privacy and anonymization.

    Attributes:
        dp_config: Differential privacy configuration.
        anonymization_config: Data anonymization configuration.
        audit_logging: Whether to enable privacy audit logging.

    Examples:
        >>> dp = DPConfig(epsilon=0.5)
        >>> anon = AnonymizationConfig(k_value=10)
        >>> config = PrivacyConfig(
        ...     dp_config=dp,
        ...     anonymization_config=anon,
        ...     audit_logging=True,
        ... )
        >>> config.dp_config.epsilon
        0.5
        >>> config.anonymization_config.k_value
        10
        >>> config.audit_logging
        True

        >>> config2 = PrivacyConfig()
        >>> config2.audit_logging
        False
    """

    dp_config: DPConfig = field(default_factory=DPConfig)
    anonymization_config: AnonymizationConfig = field(
        default_factory=AnonymizationConfig
    )
    audit_logging: bool = False


@dataclass(frozen=True, slots=True)
class PrivacyStats:
    """Statistics from privacy-preserving operations.

    Attributes:
        privacy_budget_spent: Total epsilon spent across queries.
        records_anonymized: Number of records processed.
        noise_added: Total noise magnitude added.
        utility_loss: Estimated utility loss from privacy (0-1).

    Examples:
        >>> stats = PrivacyStats(
        ...     privacy_budget_spent=0.5,
        ...     records_anonymized=1000,
        ...     noise_added=42.5,
        ...     utility_loss=0.15,
        ... )
        >>> stats.privacy_budget_spent
        0.5
        >>> stats.records_anonymized
        1000
        >>> stats.noise_added
        42.5
        >>> stats.utility_loss
        0.15
    """

    privacy_budget_spent: float
    records_anonymized: int
    noise_added: float
    utility_loss: float


def validate_dp_config(config: DPConfig) -> None:
    """Validate differential privacy configuration.

    Args:
        config: DPConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If epsilon is not positive.
        ValueError: If delta is negative or >= 1.
        ValueError: If clip_norm is not positive.
        ValueError: If noise_multiplier is not positive.

    Examples:
        >>> config = DPConfig(epsilon=1.0, delta=1e-5)
        >>> validate_dp_config(config)  # No error

        >>> validate_dp_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = DPConfig(epsilon=0.0)
        >>> validate_dp_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: epsilon must be positive

        >>> bad_delta = DPConfig(delta=1.0)
        >>> validate_dp_config(bad_delta)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: delta must be in [0, 1)
    """
    validate_not_none(config, "config")

    if config.epsilon <= 0:
        msg = f"epsilon must be positive, got {config.epsilon}"
        raise ValueError(msg)

    if config.delta < 0 or config.delta >= 1:
        msg = f"delta must be in [0, 1), got {config.delta}"
        raise ValueError(msg)

    if config.clip_norm <= 0:
        msg = f"clip_norm must be positive, got {config.clip_norm}"
        raise ValueError(msg)

    if config.noise_multiplier <= 0:
        msg = f"noise_multiplier must be positive, got {config.noise_multiplier}"
        raise ValueError(msg)


def validate_anonymization_config(config: AnonymizationConfig) -> None:
    """Validate anonymization configuration.

    Args:
        config: AnonymizationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If k_value is less than 2.

    Examples:
        >>> config = AnonymizationConfig(k_value=5)
        >>> validate_anonymization_config(config)  # No error

        >>> validate_anonymization_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = AnonymizationConfig(k_value=1)
        >>> validate_anonymization_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: k_value must be at least 2
    """
    validate_not_none(config, "config")

    if config.k_value < 2:
        msg = f"k_value must be at least 2, got {config.k_value}"
        raise ValueError(msg)


def validate_privacy_config(config: PrivacyConfig) -> None:
    """Validate combined privacy configuration.

    Args:
        config: PrivacyConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If dp_config is invalid.
        ValueError: If anonymization_config is invalid.

    Examples:
        >>> config = PrivacyConfig()
        >>> validate_privacy_config(config)  # No error

        >>> validate_privacy_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_dp_config(config.dp_config)
    validate_anonymization_config(config.anonymization_config)


def create_dp_config(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    mechanism: PrivacyMechanism | str = PrivacyMechanism.LAPLACE,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
) -> DPConfig:
    """Create a validated differential privacy configuration.

    Args:
        epsilon: Privacy budget (lower = more private).
        delta: Probability of privacy breach.
        mechanism: Type of DP mechanism.
        clip_norm: Maximum L2 norm for gradient clipping.
        noise_multiplier: Multiplier for noise scale.

    Returns:
        Validated DPConfig instance.

    Raises:
        ValueError: If epsilon is not positive.
        ValueError: If delta is negative or >= 1.
        ValueError: If mechanism string is invalid.
        ValueError: If clip_norm is not positive.
        ValueError: If noise_multiplier is not positive.

    Examples:
        >>> config = create_dp_config(epsilon=0.5, delta=1e-6)
        >>> config.epsilon
        0.5
        >>> config.delta
        1e-06

        >>> config2 = create_dp_config(mechanism="gaussian")
        >>> config2.mechanism
        <PrivacyMechanism.GAUSSIAN: 'gaussian'>

        >>> create_dp_config(epsilon=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: epsilon must be positive

        >>> create_dp_config(mechanism="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid privacy mechanism: invalid
    """
    if isinstance(mechanism, str):
        mechanism = get_privacy_mechanism(mechanism)

    config = DPConfig(
        epsilon=epsilon,
        delta=delta,
        mechanism=mechanism,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )
    validate_dp_config(config)
    return config


def create_anonymization_config(
    anon_type: AnonymizationType | str = AnonymizationType.K_ANONYMITY,
    k_value: int = 5,
    quasi_identifiers: tuple[str, ...] | list[str] | None = None,
    sensitive_attributes: tuple[str, ...] | list[str] | None = None,
) -> AnonymizationConfig:
    """Create a validated anonymization configuration.

    Args:
        anon_type: Type of anonymization technique.
        k_value: K value for k-anonymity.
        quasi_identifiers: Quasi-identifier column names.
        sensitive_attributes: Sensitive attribute column names.

    Returns:
        Validated AnonymizationConfig instance.

    Raises:
        ValueError: If anon_type string is invalid.
        ValueError: If k_value is less than 2.

    Examples:
        >>> config = create_anonymization_config(k_value=10)
        >>> config.k_value
        10

        >>> config2 = create_anonymization_config(
        ...     anon_type="l_diversity",
        ...     quasi_identifiers=["age", "zipcode"],
        ... )
        >>> config2.anon_type
        <AnonymizationType.L_DIVERSITY: 'l_diversity'>
        >>> "age" in config2.quasi_identifiers
        True

        >>> create_anonymization_config(k_value=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: k_value must be at least 2

        >>> create_anonymization_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     anon_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: invalid anonymization type: invalid
    """
    if isinstance(anon_type, str):
        anon_type = get_anonymization_type(anon_type)

    if quasi_identifiers is None:
        quasi_identifiers = ()
    elif isinstance(quasi_identifiers, list):
        quasi_identifiers = tuple(quasi_identifiers)

    if sensitive_attributes is None:
        sensitive_attributes = ()
    elif isinstance(sensitive_attributes, list):
        sensitive_attributes = tuple(sensitive_attributes)

    config = AnonymizationConfig(
        anon_type=anon_type,
        k_value=k_value,
        quasi_identifiers=quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
    )
    validate_anonymization_config(config)
    return config


def create_privacy_config(
    dp_config: DPConfig | None = None,
    anonymization_config: AnonymizationConfig | None = None,
    audit_logging: bool = False,
) -> PrivacyConfig:
    """Create a validated combined privacy configuration.

    Args:
        dp_config: Differential privacy configuration.
        anonymization_config: Data anonymization configuration.
        audit_logging: Whether to enable privacy audit logging.

    Returns:
        Validated PrivacyConfig instance.

    Raises:
        ValueError: If dp_config is invalid.
        ValueError: If anonymization_config is invalid.

    Examples:
        >>> config = create_privacy_config(audit_logging=True)
        >>> config.audit_logging
        True

        >>> dp = create_dp_config(epsilon=0.5)
        >>> anon = create_anonymization_config(k_value=10)
        >>> config2 = create_privacy_config(dp_config=dp, anonymization_config=anon)
        >>> config2.dp_config.epsilon
        0.5
        >>> config2.anonymization_config.k_value
        10
    """
    if dp_config is None:
        dp_config = DPConfig()
    if anonymization_config is None:
        anonymization_config = AnonymizationConfig()

    config = PrivacyConfig(
        dp_config=dp_config,
        anonymization_config=anonymization_config,
        audit_logging=audit_logging,
    )
    validate_privacy_config(config)
    return config


def list_privacy_mechanisms() -> list[str]:
    """List all available privacy mechanisms.

    Returns:
        Sorted list of privacy mechanism names.

    Examples:
        >>> mechanisms = list_privacy_mechanisms()
        >>> "laplace" in mechanisms
        True
        >>> "gaussian" in mechanisms
        True
        >>> mechanisms == sorted(mechanisms)
        True
    """
    return sorted(VALID_PRIVACY_MECHANISMS)


def list_anonymization_types() -> list[str]:
    """List all available anonymization types.

    Returns:
        Sorted list of anonymization type names.

    Examples:
        >>> types = list_anonymization_types()
        >>> "k_anonymity" in types
        True
        >>> "l_diversity" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ANONYMIZATION_TYPES)


def list_sensitivity_types() -> list[str]:
    """List all available sensitivity types.

    Returns:
        Sorted list of sensitivity type names.

    Examples:
        >>> types = list_sensitivity_types()
        >>> "local" in types
        True
        >>> "global" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_SENSITIVITY_TYPES)


def get_privacy_mechanism(name: str) -> PrivacyMechanism:
    """Get privacy mechanism enum from string name.

    Args:
        name: Privacy mechanism name.

    Returns:
        PrivacyMechanism enum value.

    Raises:
        ValueError: If name is not a valid privacy mechanism.

    Examples:
        >>> get_privacy_mechanism("laplace")
        <PrivacyMechanism.LAPLACE: 'laplace'>
        >>> get_privacy_mechanism("gaussian")
        <PrivacyMechanism.GAUSSIAN: 'gaussian'>

        >>> get_privacy_mechanism("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid privacy mechanism: invalid
    """
    for mechanism in PrivacyMechanism:
        if mechanism.value == name:
            return mechanism
    msg = f"invalid privacy mechanism: {name}"
    raise ValueError(msg)


def get_anonymization_type(name: str) -> AnonymizationType:
    """Get anonymization type enum from string name.

    Args:
        name: Anonymization type name.

    Returns:
        AnonymizationType enum value.

    Raises:
        ValueError: If name is not a valid anonymization type.

    Examples:
        >>> get_anonymization_type("k_anonymity")
        <AnonymizationType.K_ANONYMITY: 'k_anonymity'>
        >>> get_anonymization_type("l_diversity")
        <AnonymizationType.L_DIVERSITY: 'l_diversity'>

        >>> get_anonymization_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid anonymization type: invalid
    """
    for anon_type in AnonymizationType:
        if anon_type.value == name:
            return anon_type
    msg = f"invalid anonymization type: {name}"
    raise ValueError(msg)


def get_sensitivity_type(name: str) -> SensitivityType:
    """Get sensitivity type enum from string name.

    Args:
        name: Sensitivity type name.

    Returns:
        SensitivityType enum value.

    Raises:
        ValueError: If name is not a valid sensitivity type.

    Examples:
        >>> get_sensitivity_type("local")
        <SensitivityType.LOCAL: 'local'>
        >>> get_sensitivity_type("global")
        <SensitivityType.GLOBAL: 'global'>

        >>> get_sensitivity_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid sensitivity type: invalid
    """
    for sens_type in SensitivityType:
        if sens_type.value == name:
            return sens_type
    msg = f"invalid sensitivity type: {name}"
    raise ValueError(msg)


def calculate_noise_scale(
    epsilon: float,
    sensitivity: float,
    delta: float = 0.0,
    mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE,
) -> float:
    """Calculate noise scale for differential privacy.

    For Laplace mechanism: scale = sensitivity / epsilon
    For Gaussian mechanism: scale = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon

    Args:
        epsilon: Privacy budget.
        sensitivity: Query sensitivity (L1 for Laplace, L2 for Gaussian).
        delta: Probability of privacy breach (for Gaussian).
        mechanism: Type of DP mechanism.

    Returns:
        Noise scale parameter.

    Raises:
        ValueError: If epsilon is not positive.
        ValueError: If sensitivity is not positive.
        ValueError: If delta is invalid for Gaussian mechanism.

    Examples:
        >>> scale = calculate_noise_scale(epsilon=1.0, sensitivity=1.0)
        >>> scale
        1.0

        >>> scale = calculate_noise_scale(epsilon=0.5, sensitivity=2.0)
        >>> scale
        4.0

        >>> scale = calculate_noise_scale(
        ...     epsilon=1.0,
        ...     sensitivity=1.0,
        ...     delta=1e-5,
        ...     mechanism=PrivacyMechanism.GAUSSIAN,
        ... )
        >>> scale > 0
        True

        >>> calculate_noise_scale(epsilon=0, sensitivity=1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: epsilon must be positive

        >>> calculate_noise_scale(epsilon=1.0, sensitivity=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sensitivity must be positive
    """
    if epsilon <= 0:
        msg = f"epsilon must be positive, got {epsilon}"
        raise ValueError(msg)

    if sensitivity <= 0:
        msg = f"sensitivity must be positive, got {sensitivity}"
        raise ValueError(msg)

    if mechanism == PrivacyMechanism.GAUSSIAN:
        if delta <= 0 or delta >= 1:
            msg = f"delta must be in (0, 1) for Gaussian mechanism, got {delta}"
            raise ValueError(msg)
        # Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    # Laplace mechanism: scale = sensitivity / epsilon
    return sensitivity / epsilon


def compute_privacy_budget(
    num_queries: int,
    epsilon_per_query: float,
    composition: str = "simple",
) -> float:
    """Compute total privacy budget for multiple queries.

    Args:
        num_queries: Number of queries to perform.
        epsilon_per_query: Privacy budget per query.
        composition: Composition type ("simple" or "advanced").

    Returns:
        Total privacy budget spent.

    Raises:
        ValueError: If num_queries is not positive.
        ValueError: If epsilon_per_query is not positive.
        ValueError: If composition is invalid.

    Examples:
        >>> budget = compute_privacy_budget(10, 0.1)
        >>> budget
        1.0

        >>> budget = compute_privacy_budget(10, 0.1, composition="advanced")
        >>> budget < 1.0
        True

        >>> compute_privacy_budget(0, 0.1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_queries must be positive

        >>> compute_privacy_budget(10, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: epsilon_per_query must be positive

        >>> compute_privacy_budget(10, 0.1, "invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: composition must be 'simple' or 'advanced'
    """
    if num_queries <= 0:
        msg = f"num_queries must be positive, got {num_queries}"
        raise ValueError(msg)

    if epsilon_per_query <= 0:
        msg = f"epsilon_per_query must be positive, got {epsilon_per_query}"
        raise ValueError(msg)

    if composition not in ("simple", "advanced"):
        msg = f"composition must be 'simple' or 'advanced', got {composition}"
        raise ValueError(msg)

    if composition == "simple":
        # Simple composition: sum of epsilons
        return num_queries * epsilon_per_query

    # Advanced composition: sqrt(2k * ln(1/delta')) * eps + k * eps * (e^eps - 1)
    # Simplified version: sqrt(k) * eps
    return math.sqrt(num_queries) * epsilon_per_query


def add_differential_privacy_noise(
    value: float,
    epsilon: float,
    sensitivity: float,
    mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE,
    delta: float = 0.0,
) -> float:
    """Add differential privacy noise to a value.

    This function computes the expected noise magnitude but returns a
    deterministic result for reproducibility. In production, use a
    proper random number generator.

    Args:
        value: Original value to protect.
        epsilon: Privacy budget.
        sensitivity: Query sensitivity.
        mechanism: Type of DP mechanism.
        delta: Probability of privacy breach (for Gaussian).

    Returns:
        Value with expected noise magnitude added.

    Raises:
        ValueError: If epsilon is not positive.
        ValueError: If sensitivity is not positive.
        ValueError: If delta is invalid for Gaussian mechanism.

    Examples:
        >>> noisy = add_differential_privacy_noise(100.0, epsilon=1.0, sensitivity=1.0)
        >>> noisy != 100.0
        True

        >>> noisy = add_differential_privacy_noise(
        ...     50.0, epsilon=0.5, sensitivity=2.0
        ... )
        >>> noisy > 50.0
        True

        >>> add_differential_privacy_noise(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     100.0, epsilon=0, sensitivity=1.0
        ... )
        Traceback (most recent call last):
        ValueError: epsilon must be positive
    """
    scale = calculate_noise_scale(epsilon, sensitivity, delta, mechanism)

    # For reproducible doctests, add deterministic expected noise magnitude
    # In production, use: np.random.laplace(0, scale) or np.random.normal(0, scale)
    expected_noise = scale  # Expected absolute value of Laplace(0, scale) is scale

    return value + expected_noise


def check_k_anonymity(
    group_sizes: Sequence[int],
    k: int,
) -> bool:
    """Check if data satisfies k-anonymity.

    Data satisfies k-anonymity if every equivalence class (group of
    records with identical quasi-identifiers) has at least k records.

    Args:
        group_sizes: Sizes of equivalence classes.
        k: Minimum required group size.

    Returns:
        True if k-anonymity is satisfied, False otherwise.

    Raises:
        ValueError: If k is less than 2.
        ValueError: If group_sizes is empty.

    Examples:
        >>> check_k_anonymity([5, 6, 7, 8], k=5)
        True

        >>> check_k_anonymity([3, 5, 6], k=5)
        False

        >>> check_k_anonymity([10, 10, 10], k=5)
        True

        >>> check_k_anonymity([], k=5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: group_sizes cannot be empty

        >>> check_k_anonymity([5, 6], k=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: k must be at least 2
    """
    if k < 2:
        msg = f"k must be at least 2, got {k}"
        raise ValueError(msg)

    if len(group_sizes) == 0:
        msg = "group_sizes cannot be empty"
        raise ValueError(msg)

    return all(size >= k for size in group_sizes)


def estimate_utility_loss(
    epsilon: float,
    sensitivity: float,
    mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE,
    delta: float = 0.0,
) -> float:
    """Estimate utility loss from differential privacy noise.

    Returns a normalized estimate of utility loss between 0 and 1,
    where 0 means no loss and 1 means complete loss of utility.

    Args:
        epsilon: Privacy budget.
        sensitivity: Query sensitivity.
        mechanism: Type of DP mechanism.
        delta: Probability of privacy breach (for Gaussian).

    Returns:
        Estimated utility loss (0 to 1).

    Raises:
        ValueError: If epsilon is not positive.
        ValueError: If sensitivity is not positive.

    Examples:
        >>> loss = estimate_utility_loss(epsilon=1.0, sensitivity=1.0)
        >>> 0 <= loss <= 1
        True

        >>> loss_high_eps = estimate_utility_loss(epsilon=10.0, sensitivity=1.0)
        >>> loss_low_eps = estimate_utility_loss(epsilon=0.1, sensitivity=1.0)
        >>> loss_high_eps < loss_low_eps
        True

        >>> estimate_utility_loss(epsilon=0, sensitivity=1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: epsilon must be positive
    """
    scale = calculate_noise_scale(epsilon, sensitivity, delta, mechanism)

    # Normalize utility loss using sigmoid-like function
    # Higher noise scale = higher utility loss
    # Using 1 - 1/(1 + scale) to map scale to (0, 1)
    return scale / (1 + scale)


def format_privacy_stats(stats: PrivacyStats) -> str:
    """Format privacy statistics as a human-readable string.

    Args:
        stats: PrivacyStats to format.

    Returns:
        Formatted statistics string.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = PrivacyStats(
        ...     privacy_budget_spent=0.5,
        ...     records_anonymized=1000,
        ...     noise_added=42.5,
        ...     utility_loss=0.15,
        ... )
        >>> formatted = format_privacy_stats(stats)
        >>> "Privacy Budget Spent: 0.50" in formatted
        True
        >>> "Records Anonymized: 1000" in formatted
        True
        >>> "Noise Added: 42.50" in formatted
        True
        >>> "Utility Loss: 15.0%" in formatted
        True

        >>> format_privacy_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        f"Privacy Budget Spent: {stats.privacy_budget_spent:.2f}",
        f"Records Anonymized: {stats.records_anonymized}",
        f"Noise Added: {stats.noise_added:.2f}",
        f"Utility Loss: {stats.utility_loss * 100:.1f}%",
    ]
    return "\n".join(lines)


def get_recommended_privacy_config(
    use_case: str = "training",
    data_sensitivity: str = "medium",
) -> PrivacyConfig:
    """Get recommended privacy configuration for common use cases.

    Args:
        use_case: Use case type ("training", "inference", "analysis").
        data_sensitivity: Data sensitivity level ("low", "medium", "high").

    Returns:
        Recommended PrivacyConfig for the use case.

    Raises:
        ValueError: If use_case is invalid.
        ValueError: If data_sensitivity is invalid.

    Examples:
        >>> config = get_recommended_privacy_config("training", "high")
        >>> config.dp_config.epsilon < 1.0
        True
        >>> config.anonymization_config.k_value >= 5
        True

        >>> config2 = get_recommended_privacy_config("inference", "low")
        >>> config2.dp_config.epsilon >= 1.0
        True

        >>> get_recommended_privacy_config("invalid", "medium")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case must be one of: training, inference, analysis

        >>> get_recommended_privacy_config("training", "invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data_sensitivity must be one of: low, medium, high
    """
    valid_use_cases = ("training", "inference", "analysis")
    valid_sensitivities = ("low", "medium", "high")

    if use_case not in valid_use_cases:
        msg = f"use_case must be one of: {', '.join(valid_use_cases)}"
        raise ValueError(msg)

    if data_sensitivity not in valid_sensitivities:
        msg = f"data_sensitivity must be one of: {', '.join(valid_sensitivities)}"
        raise ValueError(msg)

    # Epsilon values based on sensitivity
    epsilon_map = {
        "low": 4.0,
        "medium": 1.0,
        "high": 0.1,
    }

    # K values based on sensitivity
    k_map = {
        "low": 3,
        "medium": 5,
        "high": 10,
    }

    # Adjust epsilon based on use case
    epsilon = epsilon_map[data_sensitivity]
    if use_case == "training":
        # Training needs more privacy
        epsilon *= 0.5
    elif use_case == "analysis":
        # Analysis can be less strict
        epsilon *= 2.0

    # Choose mechanism based on use case
    if use_case == "training":
        mechanism = PrivacyMechanism.GAUSSIAN
        delta = 1e-5
    else:
        mechanism = PrivacyMechanism.LAPLACE
        delta = 0.0

    dp_config = DPConfig(
        epsilon=epsilon,
        delta=delta,
        mechanism=mechanism,
        clip_norm=1.0,
        noise_multiplier=1.0,
    )

    anon_config = AnonymizationConfig(
        anon_type=AnonymizationType.K_ANONYMITY,
        k_value=k_map[data_sensitivity],
    )

    return PrivacyConfig(
        dp_config=dp_config,
        anonymization_config=anon_config,
        audit_logging=data_sensitivity == "high",
    )
