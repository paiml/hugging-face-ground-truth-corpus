"""Continual Learning training utilities.

This module provides functions for configuring and running continual learning
training, enabling models to learn from sequential tasks while mitigating
catastrophic forgetting through various regularization and replay strategies.

Examples:
    >>> from hf_gtc.training.continual import (
    ...     create_continual_config,
    ...     ContinualMethod,
    ... )
    >>> config = create_continual_config(method="ewc", regularization_strength=0.5)
    >>> config.method
    <ContinualMethod.EWC: 'ewc'>
    >>> config.regularization_strength
    0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ContinualMethod(Enum):
    """Supported continual learning methods.

    Attributes:
        EWC: Elastic Weight Consolidation - regularizes based on Fisher information.
        SI: Synaptic Intelligence - online importance computation.
        MAS: Memory Aware Synapses - importance from gradient magnitude.
        GEM: Gradient Episodic Memory - constrains gradients using replay buffer.
        AGEM: Averaged GEM - more efficient variant of GEM.
        REPLAY: Experience Replay - stores and replays past examples.

    Examples:
        >>> ContinualMethod.EWC.value
        'ewc'
        >>> ContinualMethod.SI.value
        'si'
        >>> ContinualMethod.REPLAY.value
        'replay'
    """

    EWC = "ewc"
    SI = "si"
    MAS = "mas"
    GEM = "gem"
    AGEM = "agem"
    REPLAY = "replay"


class ReplayStrategy(Enum):
    """Strategies for selecting samples for replay buffer.

    Attributes:
        RANDOM: Randomly sample from past data.
        RESERVOIR: Reservoir sampling for uniform distribution.
        HERDING: Select representative samples near class centroids.
        MIR: Maximally Interfered Retrieval - select most affected samples.

    Examples:
        >>> ReplayStrategy.RANDOM.value
        'random'
        >>> ReplayStrategy.RESERVOIR.value
        'reservoir'
        >>> ReplayStrategy.MIR.value
        'mir'
    """

    RANDOM = "random"
    RESERVOIR = "reservoir"
    HERDING = "herding"
    MIR = "mir"


class RegularizationType(Enum):
    """Types of regularization for weight importance.

    Attributes:
        L2: Standard L2 regularization.
        FISHER: Fisher information based regularization.
        IMPORTANCE_WEIGHTED: Custom importance weighting.

    Examples:
        >>> RegularizationType.L2.value
        'l2'
        >>> RegularizationType.FISHER.value
        'fisher'
        >>> RegularizationType.IMPORTANCE_WEIGHTED.value
        'importance_weighted'
    """

    L2 = "l2"
    FISHER = "fisher"
    IMPORTANCE_WEIGHTED = "importance_weighted"


VALID_CONTINUAL_METHODS = frozenset(m.value for m in ContinualMethod)
VALID_REPLAY_STRATEGIES = frozenset(s.value for s in ReplayStrategy)
VALID_REGULARIZATION_TYPES = frozenset(r.value for r in RegularizationType)


@dataclass(frozen=True, slots=True)
class EWCConfig:
    """Configuration for Elastic Weight Consolidation.

    Attributes:
        lambda_ewc: Regularization strength for EWC penalty.
        fisher_samples: Number of samples to estimate Fisher information.
        normalize_fisher: Whether to normalize Fisher diagonal.
        online: Whether to use online EWC variant.

    Examples:
        >>> config = EWCConfig(
        ...     lambda_ewc=1000.0,
        ...     fisher_samples=200,
        ...     normalize_fisher=True,
        ...     online=False,
        ... )
        >>> config.lambda_ewc
        1000.0
        >>> config.fisher_samples
        200
    """

    lambda_ewc: float
    fisher_samples: int
    normalize_fisher: bool
    online: bool


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Configuration for experience replay.

    Attributes:
        strategy: Strategy for selecting replay samples.
        buffer_size: Maximum number of samples in replay buffer.
        samples_per_task: Number of samples to store per task.
        prioritized: Whether to use prioritized replay.

    Examples:
        >>> config = ReplayConfig(
        ...     strategy=ReplayStrategy.RESERVOIR,
        ...     buffer_size=1000,
        ...     samples_per_task=200,
        ...     prioritized=False,
        ... )
        >>> config.buffer_size
        1000
        >>> config.strategy
        <ReplayStrategy.RESERVOIR: 'reservoir'>
    """

    strategy: ReplayStrategy
    buffer_size: int
    samples_per_task: int
    prioritized: bool


@dataclass(frozen=True, slots=True)
class ContinualConfig:
    """Main configuration for continual learning training.

    Attributes:
        method: Continual learning method to use.
        replay_config: Configuration for replay buffer (if applicable).
        regularization_strength: Strength of regularization penalty.
        task_boundary: Whether to use explicit task boundaries.

    Examples:
        >>> config = ContinualConfig(
        ...     method=ContinualMethod.EWC,
        ...     replay_config=None,
        ...     regularization_strength=0.5,
        ...     task_boundary=True,
        ... )
        >>> config.method
        <ContinualMethod.EWC: 'ewc'>
        >>> config.regularization_strength
        0.5
    """

    method: ContinualMethod
    replay_config: ReplayConfig | None
    regularization_strength: float
    task_boundary: bool


@dataclass(frozen=True, slots=True)
class ForgettingMetrics:
    """Metrics for measuring catastrophic forgetting.

    Attributes:
        backward_transfer: How much learning new tasks affects old tasks.
            Negative values indicate forgetting.
        forward_transfer: How much old knowledge helps learning new tasks.
        forgetting_rate: Average forgetting across all tasks.
        plasticity: Model's ability to learn new tasks.

    Examples:
        >>> metrics = ForgettingMetrics(
        ...     backward_transfer=-0.05,
        ...     forward_transfer=0.10,
        ...     forgetting_rate=0.03,
        ...     plasticity=0.95,
        ... )
        >>> metrics.backward_transfer
        -0.05
        >>> metrics.plasticity
        0.95
    """

    backward_transfer: float
    forward_transfer: float
    forgetting_rate: float
    plasticity: float


def validate_ewc_config(config: EWCConfig) -> None:
    """Validate EWC configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = EWCConfig(
        ...     lambda_ewc=1000.0,
        ...     fisher_samples=200,
        ...     normalize_fisher=True,
        ...     online=False,
        ... )
        >>> validate_ewc_config(config)

        >>> bad_config = EWCConfig(
        ...     lambda_ewc=-1.0,
        ...     fisher_samples=200,
        ...     normalize_fisher=True,
        ...     online=False,
        ... )
        >>> validate_ewc_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: lambda_ewc must be non-negative, got -1.0
    """
    if config.lambda_ewc < 0:
        msg = f"lambda_ewc must be non-negative, got {config.lambda_ewc}"
        raise ValueError(msg)
    if config.fisher_samples <= 0:
        msg = f"fisher_samples must be positive, got {config.fisher_samples}"
        raise ValueError(msg)


def validate_replay_config(config: ReplayConfig) -> None:
    """Validate replay configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = ReplayConfig(
        ...     strategy=ReplayStrategy.RESERVOIR,
        ...     buffer_size=1000,
        ...     samples_per_task=200,
        ...     prioritized=False,
        ... )
        >>> validate_replay_config(config)

        >>> bad_config = ReplayConfig(
        ...     strategy=ReplayStrategy.RANDOM,
        ...     buffer_size=0,
        ...     samples_per_task=200,
        ...     prioritized=False,
        ... )
        >>> validate_replay_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: buffer_size must be positive, got 0
    """
    if config.buffer_size <= 0:
        msg = f"buffer_size must be positive, got {config.buffer_size}"
        raise ValueError(msg)
    if config.samples_per_task <= 0:
        msg = f"samples_per_task must be positive, got {config.samples_per_task}"
        raise ValueError(msg)
    if config.samples_per_task > config.buffer_size:
        msg = (
            f"samples_per_task ({config.samples_per_task}) cannot exceed "
            f"buffer_size ({config.buffer_size})"
        )
        raise ValueError(msg)


def validate_continual_config(config: ContinualConfig) -> None:
    """Validate continual learning configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = ContinualConfig(
        ...     method=ContinualMethod.EWC,
        ...     replay_config=None,
        ...     regularization_strength=0.5,
        ...     task_boundary=True,
        ... )
        >>> validate_continual_config(config)

        >>> bad_config = ContinualConfig(
        ...     method=ContinualMethod.EWC,
        ...     replay_config=None,
        ...     regularization_strength=-0.1,
        ...     task_boundary=True,
        ... )
        >>> validate_continual_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: regularization_strength must be non-negative, got -0.1
    """
    if config.regularization_strength < 0:
        strength = config.regularization_strength
        msg = f"regularization_strength must be non-negative, got {strength}"
        raise ValueError(msg)
    if config.replay_config is not None:
        validate_replay_config(config.replay_config)


def validate_forgetting_metrics(metrics: ForgettingMetrics) -> None:
    """Validate forgetting metrics.

    Args:
        metrics: Metrics to validate.

    Raises:
        ValueError: If any metric value is invalid.

    Examples:
        >>> metrics = ForgettingMetrics(
        ...     backward_transfer=-0.05,
        ...     forward_transfer=0.10,
        ...     forgetting_rate=0.03,
        ...     plasticity=0.95,
        ... )
        >>> validate_forgetting_metrics(metrics)

        >>> bad_metrics = ForgettingMetrics(
        ...     backward_transfer=-0.05,
        ...     forward_transfer=0.10,
        ...     forgetting_rate=-0.1,
        ...     plasticity=0.95,
        ... )
        >>> validate_forgetting_metrics(bad_metrics)
        Traceback (most recent call last):
            ...
        ValueError: forgetting_rate must be non-negative, got -0.1
    """
    if metrics.forgetting_rate < 0:
        msg = f"forgetting_rate must be non-negative, got {metrics.forgetting_rate}"
        raise ValueError(msg)
    if not 0 <= metrics.plasticity <= 1:
        msg = f"plasticity must be between 0 and 1, got {metrics.plasticity}"
        raise ValueError(msg)


def create_ewc_config(
    lambda_ewc: float = 1000.0,
    fisher_samples: int = 200,
    normalize_fisher: bool = True,
    online: bool = False,
) -> EWCConfig:
    """Create an EWC configuration with validation.

    Args:
        lambda_ewc: Regularization strength for EWC penalty.
        fisher_samples: Number of samples to estimate Fisher information.
        normalize_fisher: Whether to normalize Fisher diagonal.
        online: Whether to use online EWC variant.

    Returns:
        Validated EWCConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_ewc_config()
        >>> config.lambda_ewc
        1000.0
        >>> config.fisher_samples
        200

        >>> config = create_ewc_config(lambda_ewc=5000.0, online=True)
        >>> config.lambda_ewc
        5000.0
        >>> config.online
        True

        >>> create_ewc_config(lambda_ewc=-1.0)
        Traceback (most recent call last):
            ...
        ValueError: lambda_ewc must be non-negative, got -1.0
    """
    config = EWCConfig(
        lambda_ewc=lambda_ewc,
        fisher_samples=fisher_samples,
        normalize_fisher=normalize_fisher,
        online=online,
    )
    validate_ewc_config(config)
    return config


def create_replay_config(
    strategy: str | ReplayStrategy = ReplayStrategy.RESERVOIR,
    buffer_size: int = 1000,
    samples_per_task: int = 200,
    prioritized: bool = False,
) -> ReplayConfig:
    """Create a replay configuration with validation.

    Args:
        strategy: Strategy for selecting replay samples.
        buffer_size: Maximum number of samples in replay buffer.
        samples_per_task: Number of samples to store per task.
        prioritized: Whether to use prioritized replay.

    Returns:
        Validated ReplayConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_replay_config()
        >>> config.buffer_size
        1000
        >>> config.strategy
        <ReplayStrategy.RESERVOIR: 'reservoir'>

        >>> config = create_replay_config(strategy="herding", buffer_size=5000)
        >>> config.strategy
        <ReplayStrategy.HERDING: 'herding'>
        >>> config.buffer_size
        5000

        >>> create_replay_config(buffer_size=0)
        Traceback (most recent call last):
            ...
        ValueError: buffer_size must be positive, got 0
    """
    if isinstance(strategy, str):
        strategy = get_replay_strategy(strategy)

    config = ReplayConfig(
        strategy=strategy,
        buffer_size=buffer_size,
        samples_per_task=samples_per_task,
        prioritized=prioritized,
    )
    validate_replay_config(config)
    return config


def create_continual_config(
    method: str | ContinualMethod = ContinualMethod.EWC,
    replay_config: ReplayConfig | None = None,
    regularization_strength: float = 0.5,
    task_boundary: bool = True,
) -> ContinualConfig:
    """Create a continual learning configuration with validation.

    Args:
        method: Continual learning method to use.
        replay_config: Configuration for replay buffer (if applicable).
        regularization_strength: Strength of regularization penalty.
        task_boundary: Whether to use explicit task boundaries.

    Returns:
        Validated ContinualConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_continual_config()
        >>> config.method
        <ContinualMethod.EWC: 'ewc'>
        >>> config.regularization_strength
        0.5

        >>> replay_config = create_continual_config(
        ...     method="replay", regularization_strength=0.0
        ... )
        >>> replay_config.method
        <ContinualMethod.REPLAY: 'replay'>

        >>> create_continual_config(regularization_strength=-0.1)
        Traceback (most recent call last):
            ...
        ValueError: regularization_strength must be non-negative, got -0.1
    """
    if isinstance(method, str):
        method = get_continual_method(method)

    config = ContinualConfig(
        method=method,
        replay_config=replay_config,
        regularization_strength=regularization_strength,
        task_boundary=task_boundary,
    )
    validate_continual_config(config)
    return config


def create_forgetting_metrics(
    backward_transfer: float = 0.0,
    forward_transfer: float = 0.0,
    forgetting_rate: float = 0.0,
    plasticity: float = 1.0,
) -> ForgettingMetrics:
    """Create forgetting metrics with validation.

    Args:
        backward_transfer: How much learning new tasks affects old tasks.
        forward_transfer: How much old knowledge helps learning new tasks.
        forgetting_rate: Average forgetting across all tasks.
        plasticity: Model's ability to learn new tasks.

    Returns:
        Validated ForgettingMetrics.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> metrics = create_forgetting_metrics()
        >>> metrics.forgetting_rate
        0.0
        >>> metrics.plasticity
        1.0

        >>> metrics = create_forgetting_metrics(
        ...     backward_transfer=-0.05,
        ...     forgetting_rate=0.03,
        ... )
        >>> metrics.backward_transfer
        -0.05

        >>> create_forgetting_metrics(forgetting_rate=-0.1)
        Traceback (most recent call last):
            ...
        ValueError: forgetting_rate must be non-negative, got -0.1
    """
    metrics = ForgettingMetrics(
        backward_transfer=backward_transfer,
        forward_transfer=forward_transfer,
        forgetting_rate=forgetting_rate,
        plasticity=plasticity,
    )
    validate_forgetting_metrics(metrics)
    return metrics


def list_continual_methods() -> list[str]:
    """List all available continual learning methods.

    Returns:
        Sorted list of continual learning method names.

    Examples:
        >>> methods = list_continual_methods()
        >>> "ewc" in methods
        True
        >>> "replay" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_CONTINUAL_METHODS)


def list_replay_strategies() -> list[str]:
    """List all available replay strategies.

    Returns:
        Sorted list of replay strategy names.

    Examples:
        >>> strategies = list_replay_strategies()
        >>> "reservoir" in strategies
        True
        >>> "random" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_REPLAY_STRATEGIES)


def list_regularization_types() -> list[str]:
    """List all available regularization types.

    Returns:
        Sorted list of regularization type names.

    Examples:
        >>> types = list_regularization_types()
        >>> "fisher" in types
        True
        >>> "l2" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_REGULARIZATION_TYPES)


def get_continual_method(name: str) -> ContinualMethod:
    """Get continual learning method enum from string name.

    Args:
        name: Name of the continual learning method.

    Returns:
        Corresponding ContinualMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_continual_method("ewc")
        <ContinualMethod.EWC: 'ewc'>
        >>> get_continual_method("replay")
        <ContinualMethod.REPLAY: 'replay'>

        >>> get_continual_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if name not in VALID_CONTINUAL_METHODS:
        msg = f"method must be one of {VALID_CONTINUAL_METHODS}, got '{name}'"
        raise ValueError(msg)
    return ContinualMethod(name)


def get_replay_strategy(name: str) -> ReplayStrategy:
    """Get replay strategy enum from string name.

    Args:
        name: Name of the replay strategy.

    Returns:
        Corresponding ReplayStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_replay_strategy("reservoir")
        <ReplayStrategy.RESERVOIR: 'reservoir'>
        >>> get_replay_strategy("random")
        <ReplayStrategy.RANDOM: 'random'>

        >>> get_replay_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: strategy must be one of ...
    """
    if name not in VALID_REPLAY_STRATEGIES:
        msg = f"strategy must be one of {VALID_REPLAY_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return ReplayStrategy(name)


def get_regularization_type(name: str) -> RegularizationType:
    """Get regularization type enum from string name.

    Args:
        name: Name of the regularization type.

    Returns:
        Corresponding RegularizationType enum.

    Raises:
        ValueError: If type name is invalid.

    Examples:
        >>> get_regularization_type("fisher")
        <RegularizationType.FISHER: 'fisher'>
        >>> get_regularization_type("l2")
        <RegularizationType.L2: 'l2'>

        >>> get_regularization_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: regularization_type must be one of ...
    """
    if name not in VALID_REGULARIZATION_TYPES:
        valid = VALID_REGULARIZATION_TYPES
        msg = f"regularization_type must be one of {valid}, got '{name}'"
        raise ValueError(msg)
    return RegularizationType(name)


def calculate_fisher_information(
    gradients: tuple[float, ...],
    num_samples: int,
) -> tuple[float, ...]:
    """Calculate Fisher information diagonal from gradients.

    The Fisher information matrix diagonal is approximated as the
    squared gradients averaged over samples.

    Args:
        gradients: Tuple of gradient values (one per parameter).
        num_samples: Number of samples used to compute gradients.

    Returns:
        Tuple of Fisher information diagonal values.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> grads = (0.1, 0.2, 0.3)
        >>> fisher = calculate_fisher_information(grads, 10)
        >>> len(fisher) == len(grads)
        True
        >>> all(f >= 0 for f in fisher)
        True

        >>> calculate_fisher_information((), 10)
        Traceback (most recent call last):
            ...
        ValueError: gradients cannot be empty

        >>> calculate_fisher_information((0.1, 0.2), 0)
        Traceback (most recent call last):
            ...
        ValueError: num_samples must be positive, got 0
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)
    if num_samples <= 0:
        msg = f"num_samples must be positive, got {num_samples}"
        raise ValueError(msg)

    return tuple(g * g / num_samples for g in gradients)


def estimate_importance_weights(
    parameter_changes: tuple[float, ...],
    learning_rates: tuple[float, ...],
) -> tuple[float, ...]:
    """Estimate parameter importance weights for Synaptic Intelligence.

    Importance is estimated based on how much parameters contribute
    to reducing the loss (path integral of gradient * parameter change).

    Args:
        parameter_changes: Change in each parameter during training.
        learning_rates: Learning rate used for each parameter update.

    Returns:
        Tuple of importance weights for each parameter.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> changes = (0.01, 0.05, 0.02)
        >>> lrs = (0.001, 0.001, 0.001)
        >>> weights = estimate_importance_weights(changes, lrs)
        >>> len(weights) == len(changes)
        True
        >>> all(w >= 0 for w in weights)
        True

        >>> estimate_importance_weights((), (0.001,))
        Traceback (most recent call last):
            ...
        ValueError: parameter_changes cannot be empty

        >>> estimate_importance_weights((0.01,), (0.001, 0.002))
        Traceback (most recent call last):
            ...
        ValueError: parameter_changes and learning_rates must have same length
    """
    if not parameter_changes:
        msg = "parameter_changes cannot be empty"
        raise ValueError(msg)
    if not learning_rates:
        msg = "learning_rates cannot be empty"
        raise ValueError(msg)
    if len(parameter_changes) != len(learning_rates):
        msg = "parameter_changes and learning_rates must have same length"
        raise ValueError(msg)

    epsilon = 1e-8
    return tuple(
        abs(delta) / (lr + epsilon)
        for delta, lr in zip(parameter_changes, learning_rates, strict=True)
    )


def calculate_forgetting_measure(
    initial_accuracies: tuple[float, ...],
    final_accuracies: tuple[float, ...],
) -> float:
    """Calculate average forgetting across tasks.

    Forgetting is measured as the average decrease in accuracy
    on previous tasks after learning new ones.

    Args:
        initial_accuracies: Accuracy on each task immediately after learning it.
        final_accuracies: Accuracy on each task at the end of training.

    Returns:
        Average forgetting rate (positive means forgetting occurred).

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> initial = (0.95, 0.90, 0.85)
        >>> final = (0.90, 0.85, 0.85)
        >>> forgetting = calculate_forgetting_measure(initial, final)
        >>> 0 <= forgetting <= 1
        True

        >>> calculate_forgetting_measure((), ())
        Traceback (most recent call last):
            ...
        ValueError: accuracies cannot be empty

        >>> calculate_forgetting_measure((0.9,), (0.8, 0.7))
        Traceback (most recent call last):
            ...
        ValueError: initial and final accuracies must have same length
    """
    if not initial_accuracies or not final_accuracies:
        msg = "accuracies cannot be empty"
        raise ValueError(msg)
    if len(initial_accuracies) != len(final_accuracies):
        msg = "initial and final accuracies must have same length"
        raise ValueError(msg)

    forgetting_values = tuple(
        max(0.0, init - final)
        for init, final in zip(initial_accuracies, final_accuracies, strict=True)
    )

    return sum(forgetting_values) / len(forgetting_values)


def calculate_plasticity_score(
    task_accuracies: tuple[float, ...],
    baseline_accuracy: float = 0.9,
) -> float:
    """Calculate model plasticity (ability to learn new tasks).

    Plasticity is measured as how well the model achieves
    expected accuracy on new tasks.

    Args:
        task_accuracies: Accuracy achieved on each task when first learned.
        baseline_accuracy: Expected accuracy for a fully plastic model.

    Returns:
        Plasticity score between 0 and 1.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> accuracies = (0.92, 0.88, 0.90)
        >>> plasticity = calculate_plasticity_score(accuracies)
        >>> 0 <= plasticity <= 1
        True

        >>> calculate_plasticity_score(())
        Traceback (most recent call last):
            ...
        ValueError: task_accuracies cannot be empty

        >>> calculate_plasticity_score((0.9,), baseline_accuracy=0.0)
        Traceback (most recent call last):
            ...
        ValueError: baseline_accuracy must be positive, got 0.0
    """
    if not task_accuracies:
        msg = "task_accuracies cannot be empty"
        raise ValueError(msg)
    if baseline_accuracy <= 0:
        msg = f"baseline_accuracy must be positive, got {baseline_accuracy}"
        raise ValueError(msg)

    avg_accuracy = sum(task_accuracies) / len(task_accuracies)
    return min(1.0, avg_accuracy / baseline_accuracy)


def format_forgetting_metrics(metrics: ForgettingMetrics) -> str:
    """Format forgetting metrics as a human-readable string.

    Args:
        metrics: Metrics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> metrics = create_forgetting_metrics(
        ...     backward_transfer=-0.05,
        ...     forward_transfer=0.10,
        ...     forgetting_rate=0.03,
        ...     plasticity=0.95,
        ... )
        >>> formatted = format_forgetting_metrics(metrics)
        >>> "Backward Transfer: -0.0500" in formatted
        True
        >>> "Plasticity: 95.0%" in formatted
        True
    """
    return (
        f"Forgetting Metrics:\n"
        f"  Backward Transfer: {metrics.backward_transfer:.4f}\n"
        f"  Forward Transfer: {metrics.forward_transfer:.4f}\n"
        f"  Forgetting Rate: {metrics.forgetting_rate:.4f}\n"
        f"  Plasticity: {metrics.plasticity * 100:.1f}%"
    )


def get_recommended_continual_config(num_tasks: int) -> ContinualConfig:
    """Get recommended continual learning configuration based on number of tasks.

    Args:
        num_tasks: Number of tasks to be learned sequentially.

    Returns:
        Recommended ContinualConfig for the given scenario.

    Raises:
        ValueError: If num_tasks is invalid.

    Examples:
        >>> config = get_recommended_continual_config(2)
        >>> config.method
        <ContinualMethod.EWC: 'ewc'>

        >>> config = get_recommended_continual_config(5)
        >>> config.method
        <ContinualMethod.SI: 'si'>

        >>> config = get_recommended_continual_config(15)
        >>> config.method
        <ContinualMethod.REPLAY: 'replay'>

        >>> get_recommended_continual_config(0)
        Traceback (most recent call last):
            ...
        ValueError: num_tasks must be positive, got 0
    """
    if num_tasks <= 0:
        msg = f"num_tasks must be positive, got {num_tasks}"
        raise ValueError(msg)

    if num_tasks <= 3:
        # Few tasks: EWC is simple and effective
        return create_continual_config(
            method=ContinualMethod.EWC,
            regularization_strength=0.5,
            task_boundary=True,
        )
    elif num_tasks <= 10:
        # Moderate tasks: SI provides better online estimation
        return create_continual_config(
            method=ContinualMethod.SI,
            regularization_strength=0.3,
            task_boundary=True,
        )
    else:
        # Many tasks: Replay-based methods scale better
        replay_config = create_replay_config(
            strategy=ReplayStrategy.RESERVOIR,
            buffer_size=min(5000, 500 * num_tasks),
            samples_per_task=min(500, 5000 // num_tasks),
            prioritized=True,
        )
        return create_continual_config(
            method=ContinualMethod.REPLAY,
            replay_config=replay_config,
            regularization_strength=0.1,
            task_boundary=False,
        )


def calculate_ewc_penalty(
    current_params: tuple[float, ...],
    optimal_params: tuple[float, ...],
    fisher_diagonal: tuple[float, ...],
    lambda_ewc: float,
) -> float:
    """Calculate EWC regularization penalty.

    The EWC penalty penalizes deviation from optimal parameters
    weighted by their Fisher information (importance).

    Args:
        current_params: Current parameter values.
        optimal_params: Optimal parameter values from previous task.
        fisher_diagonal: Fisher information diagonal values.
        lambda_ewc: Regularization strength.

    Returns:
        EWC penalty value.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> current = (0.5, 0.6, 0.7)
        >>> optimal = (0.4, 0.5, 0.6)
        >>> fisher = (1.0, 2.0, 3.0)
        >>> penalty = calculate_ewc_penalty(current, optimal, fisher, 1000.0)
        >>> penalty > 0
        True

        >>> calculate_ewc_penalty((), (0.5,), (1.0,), 1000.0)
        Traceback (most recent call last):
            ...
        ValueError: parameter tuples cannot be empty
    """
    if not current_params or not optimal_params or not fisher_diagonal:
        msg = "parameter tuples cannot be empty"
        raise ValueError(msg)
    if not (len(current_params) == len(optimal_params) == len(fisher_diagonal)):
        msg = "all parameter tuples must have same length"
        raise ValueError(msg)
    if lambda_ewc < 0:
        msg = f"lambda_ewc must be non-negative, got {lambda_ewc}"
        raise ValueError(msg)

    penalty = sum(
        f * (c - o) ** 2
        for c, o, f in zip(current_params, optimal_params, fisher_diagonal, strict=True)
    )

    return 0.5 * lambda_ewc * penalty


def calculate_backward_transfer(
    accuracy_matrix: tuple[tuple[float, ...], ...],
) -> float:
    """Calculate backward transfer from an accuracy matrix.

    Backward transfer measures how learning later tasks affects
    performance on earlier tasks.

    Args:
        accuracy_matrix: Matrix where row i column j is accuracy on task j
            after training on task i. Must be square.

    Returns:
        Average backward transfer (negative indicates forgetting).

    Raises:
        ValueError: If matrix is invalid.

    Examples:
        >>> matrix = (
        ...     (0.90, 0.00, 0.00),
        ...     (0.85, 0.88, 0.00),
        ...     (0.82, 0.85, 0.87),
        ... )
        >>> bt = calculate_backward_transfer(matrix)
        >>> bt < 0  # Some forgetting expected
        True

        >>> calculate_backward_transfer(())
        Traceback (most recent call last):
            ...
        ValueError: accuracy_matrix cannot be empty
    """
    if not accuracy_matrix:
        msg = "accuracy_matrix cannot be empty"
        raise ValueError(msg)

    n_tasks = len(accuracy_matrix)
    if n_tasks < 2:
        return 0.0

    # Validate square matrix
    for row in accuracy_matrix:
        if len(row) != n_tasks:
            msg = "accuracy_matrix must be square"
            raise ValueError(msg)

    # Calculate backward transfer
    total_bt = 0.0
    count = 0

    for j in range(n_tasks - 1):  # For each task except the last
        initial_acc = accuracy_matrix[j][j]  # Accuracy when task j was just learned
        final_acc = accuracy_matrix[n_tasks - 1][j]  # Final accuracy on task j
        total_bt += final_acc - initial_acc
        count += 1

    return total_bt / count if count > 0 else 0.0


def calculate_forward_transfer(
    accuracy_matrix: tuple[tuple[float, ...], ...],
    random_baseline: tuple[float, ...] | None = None,
) -> float:
    """Calculate forward transfer from an accuracy matrix.

    Forward transfer measures how learning earlier tasks helps
    with learning later tasks.

    Args:
        accuracy_matrix: Matrix where row i column j is accuracy on task j
            after training on task i.
        random_baseline: Random/zero-shot accuracy for each task.

    Returns:
        Average forward transfer (positive indicates positive transfer).

    Raises:
        ValueError: If matrix is invalid.

    Examples:
        >>> matrix = (
        ...     (0.90, 0.30, 0.25),
        ...     (0.85, 0.88, 0.35),
        ...     (0.82, 0.85, 0.87),
        ... )
        >>> baseline = (0.10, 0.10, 0.10)
        >>> ft = calculate_forward_transfer(matrix, baseline)
        >>> ft > 0  # Positive transfer expected
        True

        >>> calculate_forward_transfer(())
        Traceback (most recent call last):
            ...
        ValueError: accuracy_matrix cannot be empty
    """
    if not accuracy_matrix:
        msg = "accuracy_matrix cannot be empty"
        raise ValueError(msg)

    n_tasks = len(accuracy_matrix)
    if n_tasks < 2:
        return 0.0

    if random_baseline is None:
        random_baseline = tuple(0.0 for _ in range(n_tasks))

    if len(random_baseline) != n_tasks:
        msg = f"random_baseline length must match number of tasks ({n_tasks})"
        raise ValueError(msg)

    # Calculate forward transfer
    total_ft = 0.0
    count = 0

    for j in range(1, n_tasks):  # For each task except the first
        # Accuracy on task j before training on it
        zero_shot_acc = accuracy_matrix[j - 1][j]
        baseline = random_baseline[j]
        total_ft += zero_shot_acc - baseline
        count += 1

    return total_ft / count if count > 0 else 0.0
