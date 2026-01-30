"""Federated learning utilities.

This module provides functions for configuring federated learning
workflows, including client selection, update aggregation, and
privacy-preserving mechanisms.

Examples:
    >>> from hf_gtc.training.federated import (
    ...     create_federated_config,
    ...     AggregationMethod,
    ... )
    >>> config = create_federated_config(num_clients=10, rounds=5)
    >>> config.num_clients
    10
    >>> config.rounds
    5
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class AggregationMethod(Enum):
    """Methods for aggregating client model updates.

    Attributes:
        FEDAVG: Federated Averaging - weighted average by data size.
        FEDPROX: FedProx - adds proximal term to handle heterogeneity.
        SCAFFOLD: SCAFFOLD - uses control variates for variance reduction.
        FEDADAM: FedAdam - server-side adaptive optimization.

    Examples:
        >>> AggregationMethod.FEDAVG.value
        'fedavg'
        >>> AggregationMethod.FEDPROX.value
        'fedprox'
        >>> AggregationMethod.SCAFFOLD.value
        'scaffold'
    """

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDADAM = "fedadam"


class ClientSelectionStrategy(Enum):
    """Strategies for selecting clients in each round.

    Attributes:
        RANDOM: Random selection of clients each round.
        ROUND_ROBIN: Deterministic round-robin selection.
        RESOURCE_BASED: Select based on client compute capacity.
        CONTRIBUTION: Select based on past contribution quality.

    Examples:
        >>> ClientSelectionStrategy.RANDOM.value
        'random'
        >>> ClientSelectionStrategy.ROUND_ROBIN.value
        'round_robin'
        >>> ClientSelectionStrategy.RESOURCE_BASED.value
        'resource_based'
    """

    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    RESOURCE_BASED = "resource_based"
    CONTRIBUTION = "contribution"


class PrivacyMechanism(Enum):
    """Privacy-preserving mechanisms for federated learning.

    Attributes:
        NONE: No privacy mechanism applied.
        LOCAL_DP: Local differential privacy at each client.
        CENTRAL_DP: Central differential privacy at the server.
        SECURE_AGGREGATION: Cryptographic secure aggregation.

    Examples:
        >>> PrivacyMechanism.NONE.value
        'none'
        >>> PrivacyMechanism.LOCAL_DP.value
        'local_dp'
        >>> PrivacyMechanism.SECURE_AGGREGATION.value
        'secure_aggregation'
    """

    NONE = "none"
    LOCAL_DP = "local_dp"
    CENTRAL_DP = "central_dp"
    SECURE_AGGREGATION = "secure_aggregation"


VALID_AGGREGATION_METHODS = frozenset(m.value for m in AggregationMethod)
VALID_CLIENT_SELECTION_STRATEGIES = frozenset(s.value for s in ClientSelectionStrategy)
VALID_PRIVACY_MECHANISMS = frozenset(p.value for p in PrivacyMechanism)


@dataclass(frozen=True, slots=True)
class FederatedConfig:
    """Configuration for federated learning setup.

    Attributes:
        aggregation_method: Method for aggregating client updates.
        num_clients: Total number of participating clients.
        rounds: Number of communication rounds.
        local_epochs: Number of local training epochs per client.
        client_fraction: Fraction of clients to sample each round.

    Examples:
        >>> config = FederatedConfig(
        ...     aggregation_method=AggregationMethod.FEDAVG,
        ...     num_clients=100,
        ...     rounds=50,
        ...     local_epochs=5,
        ...     client_fraction=0.1,
        ... )
        >>> config.num_clients
        100
        >>> config.client_fraction
        0.1
    """

    aggregation_method: AggregationMethod
    num_clients: int
    rounds: int
    local_epochs: int
    client_fraction: float


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    """Configuration for privacy-preserving mechanisms.

    Attributes:
        mechanism: Type of privacy mechanism to apply.
        epsilon: Privacy budget (lower = more privacy).
        delta: Probability of privacy violation.
        clip_norm: Gradient clipping norm for DP.
        noise_multiplier: Noise scale for DP mechanisms.

    Examples:
        >>> config = PrivacyConfig(
        ...     mechanism=PrivacyMechanism.LOCAL_DP,
        ...     epsilon=1.0,
        ...     delta=1e-5,
        ...     clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ... )
        >>> config.epsilon
        1.0
        >>> config.noise_multiplier
        1.1
    """

    mechanism: PrivacyMechanism
    epsilon: float
    delta: float
    clip_norm: float
    noise_multiplier: float


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """Configuration for an individual federated client.

    Attributes:
        client_id: Unique identifier for the client.
        data_size: Number of data samples at this client.
        compute_capacity: Relative compute capacity (0-1).
        availability: Probability of client being available.

    Examples:
        >>> config = ClientConfig(
        ...     client_id="client_001",
        ...     data_size=1000,
        ...     compute_capacity=0.8,
        ...     availability=0.95,
        ... )
        >>> config.client_id
        'client_001'
        >>> config.data_size
        1000
    """

    client_id: str
    data_size: int
    compute_capacity: float
    availability: float


@dataclass(frozen=True, slots=True)
class FederatedStats:
    """Statistics from a federated learning round.

    Attributes:
        global_round: Current global round number.
        participating_clients: Number of clients in this round.
        aggregation_time: Time spent on aggregation (seconds).
        privacy_budget_spent: Cumulative privacy budget used.

    Examples:
        >>> stats = FederatedStats(
        ...     global_round=10,
        ...     participating_clients=20,
        ...     aggregation_time=5.5,
        ...     privacy_budget_spent=0.5,
        ... )
        >>> stats.global_round
        10
        >>> stats.participating_clients
        20
    """

    global_round: int
    participating_clients: int
    aggregation_time: float
    privacy_budget_spent: float


def validate_federated_config(config: FederatedConfig) -> None:
    """Validate federated learning configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = FederatedConfig(
        ...     aggregation_method=AggregationMethod.FEDAVG,
        ...     num_clients=100,
        ...     rounds=50,
        ...     local_epochs=5,
        ...     client_fraction=0.1,
        ... )
        >>> validate_federated_config(config)

        >>> bad_config = FederatedConfig(
        ...     aggregation_method=AggregationMethod.FEDAVG,
        ...     num_clients=0,
        ...     rounds=50,
        ...     local_epochs=5,
        ...     client_fraction=0.1,
        ... )
        >>> validate_federated_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_clients must be positive, got 0
    """
    if config.num_clients <= 0:
        msg = f"num_clients must be positive, got {config.num_clients}"
        raise ValueError(msg)
    if config.rounds <= 0:
        msg = f"rounds must be positive, got {config.rounds}"
        raise ValueError(msg)
    if config.local_epochs <= 0:
        msg = f"local_epochs must be positive, got {config.local_epochs}"
        raise ValueError(msg)
    if not 0 < config.client_fraction <= 1:
        msg = f"client_fraction must be in (0, 1], got {config.client_fraction}"
        raise ValueError(msg)


def validate_privacy_config(config: PrivacyConfig) -> None:
    """Validate privacy configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = PrivacyConfig(
        ...     mechanism=PrivacyMechanism.LOCAL_DP,
        ...     epsilon=1.0,
        ...     delta=1e-5,
        ...     clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ... )
        >>> validate_privacy_config(config)

        >>> bad_config = PrivacyConfig(
        ...     mechanism=PrivacyMechanism.LOCAL_DP,
        ...     epsilon=-1.0,
        ...     delta=1e-5,
        ...     clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ... )
        >>> validate_privacy_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: epsilon must be positive, got -1.0
    """
    if config.mechanism != PrivacyMechanism.NONE:
        if config.epsilon <= 0:
            msg = f"epsilon must be positive, got {config.epsilon}"
            raise ValueError(msg)
        if not 0 < config.delta < 1:
            msg = f"delta must be in (0, 1), got {config.delta}"
            raise ValueError(msg)
        if config.clip_norm <= 0:
            msg = f"clip_norm must be positive, got {config.clip_norm}"
            raise ValueError(msg)
        if config.noise_multiplier <= 0:
            msg = f"noise_multiplier must be positive, got {config.noise_multiplier}"
            raise ValueError(msg)


def validate_client_config(config: ClientConfig) -> None:
    """Validate client configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = ClientConfig(
        ...     client_id="client_001",
        ...     data_size=1000,
        ...     compute_capacity=0.8,
        ...     availability=0.95,
        ... )
        >>> validate_client_config(config)

        >>> bad_config = ClientConfig(
        ...     client_id="",
        ...     data_size=1000,
        ...     compute_capacity=0.8,
        ...     availability=0.95,
        ... )
        >>> validate_client_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: client_id cannot be empty
    """
    if not config.client_id or not config.client_id.strip():
        msg = "client_id cannot be empty"
        raise ValueError(msg)
    if config.data_size <= 0:
        msg = f"data_size must be positive, got {config.data_size}"
        raise ValueError(msg)
    if not 0 < config.compute_capacity <= 1:
        msg = f"compute_capacity must be in (0, 1], got {config.compute_capacity}"
        raise ValueError(msg)
    if not 0 < config.availability <= 1:
        msg = f"availability must be in (0, 1], got {config.availability}"
        raise ValueError(msg)


def validate_federated_stats(stats: FederatedStats) -> None:
    """Validate federated stats.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any value is invalid.

    Examples:
        >>> stats = FederatedStats(
        ...     global_round=10,
        ...     participating_clients=20,
        ...     aggregation_time=5.5,
        ...     privacy_budget_spent=0.5,
        ... )
        >>> validate_federated_stats(stats)

        >>> bad_stats = FederatedStats(
        ...     global_round=-1,
        ...     participating_clients=20,
        ...     aggregation_time=5.5,
        ...     privacy_budget_spent=0.5,
        ... )
        >>> validate_federated_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: global_round must be non-negative, got -1
    """
    if stats.global_round < 0:
        msg = f"global_round must be non-negative, got {stats.global_round}"
        raise ValueError(msg)
    if stats.participating_clients < 0:
        msg = (
            f"participating_clients must be non-negative, "
            f"got {stats.participating_clients}"
        )
        raise ValueError(msg)
    if stats.aggregation_time < 0:
        msg = f"aggregation_time must be non-negative, got {stats.aggregation_time}"
        raise ValueError(msg)
    if stats.privacy_budget_spent < 0:
        msg = (
            f"privacy_budget_spent must be non-negative, "
            f"got {stats.privacy_budget_spent}"
        )
        raise ValueError(msg)


def create_federated_config(
    aggregation_method: str | AggregationMethod = AggregationMethod.FEDAVG,
    num_clients: int = 10,
    rounds: int = 10,
    local_epochs: int = 1,
    client_fraction: float = 0.1,
) -> FederatedConfig:
    """Create a federated learning configuration with validation.

    Args:
        aggregation_method: Method for aggregating updates.
        num_clients: Total number of clients.
        rounds: Number of communication rounds.
        local_epochs: Local training epochs per round.
        client_fraction: Fraction of clients to sample.

    Returns:
        Validated FederatedConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_federated_config()
        >>> config.num_clients
        10
        >>> config.rounds
        10

        >>> config = create_federated_config(
        ...     aggregation_method="fedprox",
        ...     num_clients=100,
        ...     client_fraction=0.2,
        ... )
        >>> config.aggregation_method
        <AggregationMethod.FEDPROX: 'fedprox'>
        >>> config.client_fraction
        0.2

        >>> create_federated_config(num_clients=0)
        Traceback (most recent call last):
            ...
        ValueError: num_clients must be positive, got 0
    """
    if isinstance(aggregation_method, str):
        aggregation_method = get_aggregation_method(aggregation_method)

    config = FederatedConfig(
        aggregation_method=aggregation_method,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
    )
    validate_federated_config(config)
    return config


def create_privacy_config(
    mechanism: str | PrivacyMechanism = PrivacyMechanism.NONE,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1,
) -> PrivacyConfig:
    """Create a privacy configuration with validation.

    Args:
        mechanism: Type of privacy mechanism.
        epsilon: Privacy budget.
        delta: Privacy violation probability.
        clip_norm: Gradient clipping norm.
        noise_multiplier: Noise scale for DP.

    Returns:
        Validated PrivacyConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_privacy_config()
        >>> config.mechanism
        <PrivacyMechanism.NONE: 'none'>

        >>> config = create_privacy_config(
        ...     mechanism="local_dp",
        ...     epsilon=0.5,
        ...     noise_multiplier=1.5,
        ... )
        >>> config.mechanism
        <PrivacyMechanism.LOCAL_DP: 'local_dp'>
        >>> config.epsilon
        0.5

        >>> create_privacy_config(mechanism="local_dp", epsilon=-1.0)
        Traceback (most recent call last):
            ...
        ValueError: epsilon must be positive, got -1.0
    """
    if isinstance(mechanism, str):
        mechanism = get_privacy_mechanism(mechanism)

    config = PrivacyConfig(
        mechanism=mechanism,
        epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )
    validate_privacy_config(config)
    return config


def create_client_config(
    client_id: str,
    data_size: int = 1000,
    compute_capacity: float = 1.0,
    availability: float = 1.0,
) -> ClientConfig:
    """Create a client configuration with validation.

    Args:
        client_id: Unique identifier for the client.
        data_size: Number of data samples.
        compute_capacity: Relative compute capacity.
        availability: Probability of availability.

    Returns:
        Validated ClientConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_client_config("client_001")
        >>> config.client_id
        'client_001'
        >>> config.data_size
        1000

        >>> config = create_client_config(
        ...     "client_002",
        ...     data_size=5000,
        ...     compute_capacity=0.5,
        ... )
        >>> config.data_size
        5000
        >>> config.compute_capacity
        0.5

        >>> create_client_config("")
        Traceback (most recent call last):
            ...
        ValueError: client_id cannot be empty
    """
    config = ClientConfig(
        client_id=client_id,
        data_size=data_size,
        compute_capacity=compute_capacity,
        availability=availability,
    )
    validate_client_config(config)
    return config


def create_federated_stats(
    global_round: int = 0,
    participating_clients: int = 0,
    aggregation_time: float = 0.0,
    privacy_budget_spent: float = 0.0,
) -> FederatedStats:
    """Create federated learning statistics.

    Args:
        global_round: Current global round number.
        participating_clients: Number of participating clients.
        aggregation_time: Time for aggregation in seconds.
        privacy_budget_spent: Cumulative privacy budget used.

    Returns:
        Validated FederatedStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_federated_stats()
        >>> stats.global_round
        0

        >>> stats = create_federated_stats(
        ...     global_round=5,
        ...     participating_clients=10,
        ...     aggregation_time=2.5,
        ... )
        >>> stats.global_round
        5
        >>> stats.participating_clients
        10

        >>> create_federated_stats(global_round=-1)
        Traceback (most recent call last):
            ...
        ValueError: global_round must be non-negative, got -1
    """
    stats = FederatedStats(
        global_round=global_round,
        participating_clients=participating_clients,
        aggregation_time=aggregation_time,
        privacy_budget_spent=privacy_budget_spent,
    )
    validate_federated_stats(stats)
    return stats


def list_aggregation_methods() -> list[str]:
    """List all available aggregation methods.

    Returns:
        Sorted list of aggregation method names.

    Examples:
        >>> methods = list_aggregation_methods()
        >>> "fedavg" in methods
        True
        >>> "fedprox" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_AGGREGATION_METHODS)


def list_client_selection_strategies() -> list[str]:
    """List all available client selection strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_client_selection_strategies()
        >>> "random" in strategies
        True
        >>> "round_robin" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_CLIENT_SELECTION_STRATEGIES)


def list_privacy_mechanisms() -> list[str]:
    """List all available privacy mechanisms.

    Returns:
        Sorted list of privacy mechanism names.

    Examples:
        >>> mechanisms = list_privacy_mechanisms()
        >>> "none" in mechanisms
        True
        >>> "local_dp" in mechanisms
        True
        >>> mechanisms == sorted(mechanisms)
        True
    """
    return sorted(VALID_PRIVACY_MECHANISMS)


def get_aggregation_method(name: str) -> AggregationMethod:
    """Get aggregation method enum from string name.

    Args:
        name: Name of the aggregation method.

    Returns:
        Corresponding AggregationMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_aggregation_method("fedavg")
        <AggregationMethod.FEDAVG: 'fedavg'>
        >>> get_aggregation_method("fedprox")
        <AggregationMethod.FEDPROX: 'fedprox'>

        >>> get_aggregation_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: aggregation_method must be one of ...
    """
    if name not in VALID_AGGREGATION_METHODS:
        msg = (
            f"aggregation_method must be one of {VALID_AGGREGATION_METHODS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return AggregationMethod(name)


def get_client_selection_strategy(name: str) -> ClientSelectionStrategy:
    """Get client selection strategy enum from string name.

    Args:
        name: Name of the selection strategy.

    Returns:
        Corresponding ClientSelectionStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_client_selection_strategy("random")
        <ClientSelectionStrategy.RANDOM: 'random'>
        >>> get_client_selection_strategy("round_robin")
        <ClientSelectionStrategy.ROUND_ROBIN: 'round_robin'>

        >>> get_client_selection_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: client_selection_strategy must be one of ...
    """
    if name not in VALID_CLIENT_SELECTION_STRATEGIES:
        msg = (
            f"client_selection_strategy must be one of "
            f"{VALID_CLIENT_SELECTION_STRATEGIES}, got '{name}'"
        )
        raise ValueError(msg)
    return ClientSelectionStrategy(name)


def get_privacy_mechanism(name: str) -> PrivacyMechanism:
    """Get privacy mechanism enum from string name.

    Args:
        name: Name of the privacy mechanism.

    Returns:
        Corresponding PrivacyMechanism enum.

    Raises:
        ValueError: If mechanism name is invalid.

    Examples:
        >>> get_privacy_mechanism("none")
        <PrivacyMechanism.NONE: 'none'>
        >>> get_privacy_mechanism("local_dp")
        <PrivacyMechanism.LOCAL_DP: 'local_dp'>

        >>> get_privacy_mechanism("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: privacy_mechanism must be one of ...
    """
    if name not in VALID_PRIVACY_MECHANISMS:
        msg = (
            f"privacy_mechanism must be one of {VALID_PRIVACY_MECHANISMS}, got '{name}'"
        )
        raise ValueError(msg)
    return PrivacyMechanism(name)


def calculate_client_weights(
    clients: tuple[ClientConfig, ...],
) -> tuple[float, ...]:
    """Calculate aggregation weights for clients based on data size.

    Args:
        clients: Tuple of client configurations.

    Returns:
        Tuple of normalized weights (sum to 1.0).

    Raises:
        ValueError: If clients tuple is empty.

    Examples:
        >>> c1 = create_client_config("c1", data_size=1000)
        >>> c2 = create_client_config("c2", data_size=3000)
        >>> weights = calculate_client_weights((c1, c2))
        >>> weights
        (0.25, 0.75)

        >>> c1 = create_client_config("c1", data_size=100)
        >>> c2 = create_client_config("c2", data_size=100)
        >>> c3 = create_client_config("c3", data_size=100)
        >>> weights = calculate_client_weights((c1, c2, c3))
        >>> sum(weights)
        1.0

        >>> calculate_client_weights(())
        Traceback (most recent call last):
            ...
        ValueError: clients cannot be empty
    """
    if not clients:
        msg = "clients cannot be empty"
        raise ValueError(msg)

    total_data = sum(c.data_size for c in clients)
    return tuple(c.data_size / total_data for c in clients)


def select_clients(
    clients: tuple[ClientConfig, ...],
    num_to_select: int,
    strategy: str | ClientSelectionStrategy = ClientSelectionStrategy.RANDOM,
    round_number: int = 0,
    seed: int | None = None,
) -> tuple[ClientConfig, ...]:
    """Select clients for a federated learning round.

    Args:
        clients: Tuple of all available clients.
        num_to_select: Number of clients to select.
        strategy: Selection strategy to use.
        round_number: Current round (for round_robin strategy).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of selected client configurations.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> c1 = create_client_config("c1", data_size=1000, compute_capacity=0.5)
        >>> c2 = create_client_config("c2", data_size=2000, compute_capacity=0.8)
        >>> c3 = create_client_config("c3", data_size=1500, compute_capacity=1.0)
        >>> selected = select_clients((c1, c2, c3), 2, "random", seed=42)
        >>> len(selected)
        2

        >>> selected = select_clients((c1, c2, c3), 2, "resource_based")
        >>> selected[0].compute_capacity >= selected[1].compute_capacity
        True

        >>> select_clients((), 2, "random")
        Traceback (most recent call last):
            ...
        ValueError: clients cannot be empty

        >>> select_clients((c1,), 5, "random")
        Traceback (most recent call last):
            ...
        ValueError: num_to_select (5) cannot exceed number of clients (1)
    """
    if not clients:
        msg = "clients cannot be empty"
        raise ValueError(msg)
    if num_to_select <= 0:
        msg = f"num_to_select must be positive, got {num_to_select}"
        raise ValueError(msg)
    if num_to_select > len(clients):
        msg = (
            f"num_to_select ({num_to_select}) cannot exceed "
            f"number of clients ({len(clients)})"
        )
        raise ValueError(msg)

    if isinstance(strategy, str):
        strategy = get_client_selection_strategy(strategy)

    if strategy == ClientSelectionStrategy.RANDOM:
        rng = random.Random(seed)
        indices = rng.sample(range(len(clients)), num_to_select)
        return tuple(clients[i] for i in sorted(indices))

    elif strategy == ClientSelectionStrategy.ROUND_ROBIN:
        start_idx = (round_number * num_to_select) % len(clients)
        indices = [(start_idx + i) % len(clients) for i in range(num_to_select)]
        return tuple(clients[i] for i in indices)

    elif strategy == ClientSelectionStrategy.RESOURCE_BASED:
        sorted_clients = sorted(clients, key=lambda c: c.compute_capacity, reverse=True)
        return tuple(sorted_clients[:num_to_select])

    else:  # CONTRIBUTION - for now, same as random but could be extended
        rng = random.Random(seed)
        indices = rng.sample(range(len(clients)), num_to_select)
        return tuple(clients[i] for i in sorted(indices))


def aggregate_updates(
    client_updates: tuple[tuple[float, ...], ...],
    weights: tuple[float, ...],
    method: str | AggregationMethod = AggregationMethod.FEDAVG,
    proximal_mu: float = 0.01,
) -> tuple[float, ...]:
    """Aggregate client model updates into global update.

    Args:
        client_updates: Tuple of update vectors from each client.
        weights: Aggregation weights for each client.
        method: Aggregation method to use.
        proximal_mu: Proximal term coefficient (for FedProx).

    Returns:
        Aggregated global update vector.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> updates = ((1.0, 2.0), (3.0, 4.0))
        >>> weights = (0.5, 0.5)
        >>> result = aggregate_updates(updates, weights)
        >>> result
        (2.0, 3.0)

        >>> updates = ((1.0, 2.0), (2.0, 4.0))
        >>> weights = (0.25, 0.75)
        >>> result = aggregate_updates(updates, weights)
        >>> result
        (1.75, 3.5)

        >>> aggregate_updates((), (0.5,))
        Traceback (most recent call last):
            ...
        ValueError: client_updates cannot be empty

        >>> aggregate_updates(((1.0,),), (0.5, 0.5))
        Traceback (most recent call last):
            ...
        ValueError: number of updates (1) must match number of weights (2)
    """
    if not client_updates:
        msg = "client_updates cannot be empty"
        raise ValueError(msg)
    if not weights:
        msg = "weights cannot be empty"
        raise ValueError(msg)
    if len(client_updates) != len(weights):
        msg = (
            f"number of updates ({len(client_updates)}) "
            f"must match number of weights ({len(weights)})"
        )
        raise ValueError(msg)

    update_lengths = [len(u) for u in client_updates]
    if len(set(update_lengths)) > 1:
        msg = "all client updates must have the same length"
        raise ValueError(msg)

    if isinstance(method, str):
        method = get_aggregation_method(method)

    dim = len(client_updates[0])
    aggregated = [0.0] * dim

    for update, weight in zip(client_updates, weights, strict=True):
        for i, val in enumerate(update):
            aggregated[i] += weight * val

    # FedProx adds a proximal term during training, not aggregation
    # So the aggregation is the same as FedAvg
    # SCAFFOLD uses control variates which would require additional state
    # FedAdam applies adaptive optimization server-side

    return tuple(aggregated)


def calculate_privacy_budget(
    config: PrivacyConfig,
    num_rounds: int,
    sample_rate: float,
) -> float:
    """Calculate total privacy budget spent over training.

    Uses the moments accountant approximation for DP-SGD.

    Args:
        config: Privacy configuration.
        num_rounds: Number of training rounds.
        sample_rate: Fraction of data sampled per round.

    Returns:
        Total epsilon privacy budget spent.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_privacy_config(
        ...     mechanism="local_dp",
        ...     epsilon=1.0,
        ...     noise_multiplier=1.1,
        ... )
        >>> budget = calculate_privacy_budget(config, 10, 0.1)
        >>> budget > 0
        True

        >>> config = create_privacy_config(mechanism="none")
        >>> calculate_privacy_budget(config, 10, 0.1)
        0.0

        >>> config = create_privacy_config(mechanism="local_dp")
        >>> calculate_privacy_budget(config, 0, 0.1)
        Traceback (most recent call last):
            ...
        ValueError: num_rounds must be positive, got 0
    """
    if config.mechanism == PrivacyMechanism.NONE:
        return 0.0

    if num_rounds <= 0:
        msg = f"num_rounds must be positive, got {num_rounds}"
        raise ValueError(msg)
    if not 0 < sample_rate <= 1:
        msg = f"sample_rate must be in (0, 1], got {sample_rate}"
        raise ValueError(msg)

    # Simplified RDP to (epsilon, delta)-DP conversion
    # Based on the moments accountant
    sigma = config.noise_multiplier
    delta = config.delta

    # Gaussian mechanism privacy loss per round
    # Using the formula: epsilon = sqrt(2 * ln(1.25/delta)) / sigma
    per_round_eps = math.sqrt(2 * math.log(1.25 / delta)) / sigma

    # Composition over rounds with subsampling
    # Simplified advanced composition
    total_eps = per_round_eps * math.sqrt(num_rounds * sample_rate)

    return min(total_eps, config.epsilon * num_rounds)


def estimate_convergence_rounds(
    num_clients: int,
    client_fraction: float,
    data_heterogeneity: float = 0.5,
    target_accuracy: float = 0.9,
) -> int:
    """Estimate number of rounds needed for convergence.

    Args:
        num_clients: Total number of clients.
        client_fraction: Fraction of clients per round.
        data_heterogeneity: Degree of data heterogeneity (0-1).
        target_accuracy: Target accuracy to reach.

    Returns:
        Estimated number of rounds.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> rounds = estimate_convergence_rounds(100, 0.1)
        >>> rounds > 0
        True

        >>> rounds_hetero = estimate_convergence_rounds(
        ...     100, 0.1, data_heterogeneity=0.9
        ... )
        >>> rounds_homo = estimate_convergence_rounds(
        ...     100, 0.1, data_heterogeneity=0.1
        ... )
        >>> rounds_hetero > rounds_homo
        True

        >>> estimate_convergence_rounds(0, 0.1)
        Traceback (most recent call last):
            ...
        ValueError: num_clients must be positive, got 0
    """
    if num_clients <= 0:
        msg = f"num_clients must be positive, got {num_clients}"
        raise ValueError(msg)
    if not 0 < client_fraction <= 1:
        msg = f"client_fraction must be in (0, 1], got {client_fraction}"
        raise ValueError(msg)
    if not 0 <= data_heterogeneity <= 1:
        msg = f"data_heterogeneity must be in [0, 1], got {data_heterogeneity}"
        raise ValueError(msg)
    if not 0 < target_accuracy <= 1:
        msg = f"target_accuracy must be in (0, 1], got {target_accuracy}"
        raise ValueError(msg)

    # Base rounds for IID setting
    base_rounds = 50

    # Adjust for client participation
    participation_factor = 1 / client_fraction

    # Adjust for heterogeneity (more heterogeneous = more rounds)
    heterogeneity_factor = 1 + 2 * data_heterogeneity

    # Adjust for target accuracy (higher target = more rounds)
    accuracy_factor = 1 + math.log(1 / (1 - target_accuracy + 1e-10))

    estimated = (
        base_rounds * participation_factor * heterogeneity_factor * accuracy_factor
    )
    return max(1, int(estimated / 10))  # Scale down for practical estimates


def format_federated_stats(stats: FederatedStats) -> str:
    """Format federated statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_federated_stats(
        ...     global_round=10,
        ...     participating_clients=20,
        ...     aggregation_time=5.5,
        ...     privacy_budget_spent=0.5,
        ... )
        >>> formatted = format_federated_stats(stats)
        >>> "Round: 10" in formatted
        True
        >>> "Clients: 20" in formatted
        True
    """
    return (
        f"Federated Stats:\n"
        f"  Round: {stats.global_round}\n"
        f"  Clients: {stats.participating_clients}\n"
        f"  Aggregation Time: {stats.aggregation_time:.2f}s\n"
        f"  Privacy Budget Spent: {stats.privacy_budget_spent:.4f}"
    )


def get_recommended_federated_config(
    num_clients: int,
    privacy_required: bool = False,
) -> tuple[FederatedConfig, PrivacyConfig]:
    """Get recommended federated and privacy configurations.

    Args:
        num_clients: Number of participating clients.
        privacy_required: Whether privacy guarantees are needed.

    Returns:
        Tuple of (FederatedConfig, PrivacyConfig).

    Raises:
        ValueError: If num_clients is invalid.

    Examples:
        >>> fed_cfg, priv_cfg = get_recommended_federated_config(100)
        >>> fed_cfg.num_clients
        100
        >>> priv_cfg.mechanism
        <PrivacyMechanism.NONE: 'none'>

        >>> fed_cfg, priv_cfg = get_recommended_federated_config(
        ...     100, privacy_required=True
        ... )
        >>> priv_cfg.mechanism
        <PrivacyMechanism.LOCAL_DP: 'local_dp'>

        >>> get_recommended_federated_config(0)
        Traceback (most recent call last):
            ...
        ValueError: num_clients must be positive, got 0
    """
    if num_clients <= 0:
        msg = f"num_clients must be positive, got {num_clients}"
        raise ValueError(msg)

    # Determine client fraction based on number of clients
    if num_clients <= 10:
        client_fraction = 1.0
    elif num_clients <= 100:
        client_fraction = 0.1
    else:
        client_fraction = max(0.01, 10 / num_clients)

    # Use FedAvg for small deployments, FedProx for larger ones
    if num_clients <= 50:
        method = AggregationMethod.FEDAVG
    else:
        method = AggregationMethod.FEDPROX

    fed_config = create_federated_config(
        aggregation_method=method,
        num_clients=num_clients,
        rounds=100,
        local_epochs=5,
        client_fraction=client_fraction,
    )

    if privacy_required:
        priv_config = create_privacy_config(
            mechanism=PrivacyMechanism.LOCAL_DP,
            epsilon=8.0,
            delta=1e-5,
            clip_norm=1.0,
            noise_multiplier=1.1,
        )
    else:
        priv_config = create_privacy_config(mechanism=PrivacyMechanism.NONE)

    return fed_config, priv_config
