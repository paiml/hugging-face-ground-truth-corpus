"""Meta-learning and few-shot learning utilities.

This module provides functions for configuring and running meta-learning
training, enabling models to quickly adapt to new tasks with few examples
through methods like MAML, Prototypical Networks, and Reptile.

Examples:
    >>> from hf_gtc.training.meta_learning import (
    ...     create_meta_learning_config,
    ...     MetaLearningMethod,
    ... )
    >>> config = create_meta_learning_config(method="maml")
    >>> config.method
    <MetaLearningMethod.MAML: 'maml'>
    >>> config.episode_config.n_way
    5
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class MetaLearningMethod(Enum):
    """Supported meta-learning methods.

    Attributes:
        MAML: Model-Agnostic Meta-Learning - learns good initialization.
        PROTONET: Prototypical Networks - learns metric space for classification.
        MATCHING_NET: Matching Networks - attention-based few-shot classification.
        RELATION_NET: Relation Networks - learns a relation comparison function.
        REPTILE: First-order approximation of MAML, more scalable.

    Examples:
        >>> MetaLearningMethod.MAML.value
        'maml'
        >>> MetaLearningMethod.PROTONET.value
        'protonet'
        >>> MetaLearningMethod.REPTILE.value
        'reptile'
    """

    MAML = "maml"
    PROTONET = "protonet"
    MATCHING_NET = "matching_net"
    RELATION_NET = "relation_net"
    REPTILE = "reptile"


class TaskDistribution(Enum):
    """Task distribution strategies for meta-learning.

    Attributes:
        UNIFORM: Sample tasks uniformly at random.
        WEIGHTED: Weight tasks by difficulty or importance.
        CURRICULUM: Progressive task difficulty (easy to hard).

    Examples:
        >>> TaskDistribution.UNIFORM.value
        'uniform'
        >>> TaskDistribution.WEIGHTED.value
        'weighted'
        >>> TaskDistribution.CURRICULUM.value
        'curriculum'
    """

    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    CURRICULUM = "curriculum"


class AdaptationStrategy(Enum):
    """Adaptation strategies for meta-learning.

    Attributes:
        GRADIENT: Gradient-based adaptation (MAML, Reptile).
        METRIC: Metric-based adaptation (ProtoNet, Matching).
        HYBRID: Combination of gradient and metric approaches.

    Examples:
        >>> AdaptationStrategy.GRADIENT.value
        'gradient'
        >>> AdaptationStrategy.METRIC.value
        'metric'
        >>> AdaptationStrategy.HYBRID.value
        'hybrid'
    """

    GRADIENT = "gradient"
    METRIC = "metric"
    HYBRID = "hybrid"


VALID_META_LEARNING_METHODS = frozenset(m.value for m in MetaLearningMethod)
VALID_TASK_DISTRIBUTIONS = frozenset(d.value for d in TaskDistribution)
VALID_ADAPTATION_STRATEGIES = frozenset(s.value for s in AdaptationStrategy)


@dataclass(frozen=True, slots=True)
class MAMLConfig:
    """Configuration for Model-Agnostic Meta-Learning.

    Attributes:
        inner_lr: Learning rate for task-specific adaptation.
        outer_lr: Learning rate for meta-parameter updates.
        inner_steps: Number of gradient steps for adaptation.
        first_order: Use first-order approximation (faster but less accurate).

    Examples:
        >>> config = MAMLConfig(
        ...     inner_lr=0.01,
        ...     outer_lr=0.001,
        ...     inner_steps=5,
        ...     first_order=False,
        ... )
        >>> config.inner_lr
        0.01
        >>> config.inner_steps
        5
    """

    inner_lr: float
    outer_lr: float
    inner_steps: int
    first_order: bool


@dataclass(frozen=True, slots=True)
class ProtoNetConfig:
    """Configuration for Prototypical Networks.

    Attributes:
        distance_metric: Distance metric to use (euclidean, cosine).
        embedding_dim: Dimension of the embedding space.
        normalize: Whether to L2-normalize embeddings.

    Examples:
        >>> config = ProtoNetConfig(
        ...     distance_metric="euclidean",
        ...     embedding_dim=64,
        ...     normalize=True,
        ... )
        >>> config.distance_metric
        'euclidean'
        >>> config.embedding_dim
        64
    """

    distance_metric: str
    embedding_dim: int
    normalize: bool


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Configuration for episodic training.

    Attributes:
        n_way: Number of classes per episode.
        k_shot: Number of support examples per class.
        query_size: Number of query examples per class.

    Examples:
        >>> config = EpisodeConfig(
        ...     n_way=5,
        ...     k_shot=1,
        ...     query_size=15,
        ... )
        >>> config.n_way
        5
        >>> config.k_shot
        1
        >>> config.query_size
        15
    """

    n_way: int
    k_shot: int
    query_size: int


@dataclass(frozen=True, slots=True)
class MetaLearningConfig:
    """Main configuration for meta-learning training.

    Attributes:
        method: Meta-learning method to use.
        maml_config: Configuration for MAML (if applicable).
        protonet_config: Configuration for ProtoNet (if applicable).
        episode_config: Configuration for episodic training.

    Examples:
        >>> episode = EpisodeConfig(n_way=5, k_shot=1, query_size=15)
        >>> maml_cfg = MAMLConfig(0.01, 0.001, 5, False)
        >>> config = MetaLearningConfig(
        ...     method=MetaLearningMethod.MAML,
        ...     maml_config=maml_cfg,
        ...     protonet_config=None,
        ...     episode_config=episode,
        ... )
        >>> config.method
        <MetaLearningMethod.MAML: 'maml'>
        >>> config.episode_config.n_way
        5
    """

    method: MetaLearningMethod
    maml_config: MAMLConfig | None
    protonet_config: ProtoNetConfig | None
    episode_config: EpisodeConfig


@dataclass(frozen=True, slots=True)
class MetaLearningStats:
    """Statistics from meta-learning training.

    Attributes:
        meta_train_accuracy: Accuracy on meta-training tasks after adaptation.
        meta_test_accuracy: Accuracy on meta-test tasks after adaptation.
        adaptation_steps: Average steps needed for adaptation.
        generalization_gap: Gap between meta-train and meta-test accuracy.

    Examples:
        >>> stats = MetaLearningStats(
        ...     meta_train_accuracy=0.85,
        ...     meta_test_accuracy=0.75,
        ...     adaptation_steps=5.0,
        ...     generalization_gap=0.10,
        ... )
        >>> stats.meta_train_accuracy
        0.85
        >>> stats.generalization_gap
        0.1
    """

    meta_train_accuracy: float
    meta_test_accuracy: float
    adaptation_steps: float
    generalization_gap: float


def validate_maml_config(config: MAMLConfig) -> None:
    """Validate MAML configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = MAMLConfig(
        ...     inner_lr=0.01,
        ...     outer_lr=0.001,
        ...     inner_steps=5,
        ...     first_order=False,
        ... )
        >>> validate_maml_config(config)

        >>> bad_config = MAMLConfig(
        ...     inner_lr=-0.01,
        ...     outer_lr=0.001,
        ...     inner_steps=5,
        ...     first_order=False,
        ... )
        >>> validate_maml_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: inner_lr must be positive, got -0.01
    """
    if config.inner_lr <= 0:
        msg = f"inner_lr must be positive, got {config.inner_lr}"
        raise ValueError(msg)
    if config.outer_lr <= 0:
        msg = f"outer_lr must be positive, got {config.outer_lr}"
        raise ValueError(msg)
    if config.inner_steps <= 0:
        msg = f"inner_steps must be positive, got {config.inner_steps}"
        raise ValueError(msg)


def validate_protonet_config(config: ProtoNetConfig) -> None:
    """Validate ProtoNet configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = ProtoNetConfig(
        ...     distance_metric="euclidean",
        ...     embedding_dim=64,
        ...     normalize=True,
        ... )
        >>> validate_protonet_config(config)

        >>> bad_config = ProtoNetConfig(
        ...     distance_metric="invalid",
        ...     embedding_dim=64,
        ...     normalize=True,
        ... )
        >>> validate_protonet_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: distance_metric must be 'euclidean' or 'cosine', got 'invalid'
    """
    valid_metrics = {"euclidean", "cosine"}
    if config.distance_metric not in valid_metrics:
        msg = (
            f"distance_metric must be 'euclidean' or 'cosine', "
            f"got '{config.distance_metric}'"
        )
        raise ValueError(msg)
    if config.embedding_dim <= 0:
        msg = f"embedding_dim must be positive, got {config.embedding_dim}"
        raise ValueError(msg)


def validate_episode_config(config: EpisodeConfig) -> None:
    """Validate episode configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = EpisodeConfig(n_way=5, k_shot=1, query_size=15)
        >>> validate_episode_config(config)

        >>> bad_config = EpisodeConfig(n_way=0, k_shot=1, query_size=15)
        >>> validate_episode_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: n_way must be positive, got 0
    """
    if config.n_way <= 0:
        msg = f"n_way must be positive, got {config.n_way}"
        raise ValueError(msg)
    if config.k_shot <= 0:
        msg = f"k_shot must be positive, got {config.k_shot}"
        raise ValueError(msg)
    if config.query_size <= 0:
        msg = f"query_size must be positive, got {config.query_size}"
        raise ValueError(msg)


def validate_meta_learning_config(config: MetaLearningConfig) -> None:
    """Validate meta-learning configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> episode = EpisodeConfig(n_way=5, k_shot=1, query_size=15)
        >>> config = MetaLearningConfig(
        ...     method=MetaLearningMethod.PROTONET,
        ...     maml_config=None,
        ...     protonet_config=ProtoNetConfig("euclidean", 64, True),
        ...     episode_config=episode,
        ... )
        >>> validate_meta_learning_config(config)

        >>> bad_episode = EpisodeConfig(n_way=-1, k_shot=1, query_size=15)
        >>> bad_config = MetaLearningConfig(
        ...     method=MetaLearningMethod.PROTONET,
        ...     maml_config=None,
        ...     protonet_config=None,
        ...     episode_config=bad_episode,
        ... )
        >>> validate_meta_learning_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: n_way must be positive, got -1
    """
    validate_episode_config(config.episode_config)
    if config.maml_config is not None:
        validate_maml_config(config.maml_config)
    if config.protonet_config is not None:
        validate_protonet_config(config.protonet_config)


def validate_meta_learning_stats(stats: MetaLearningStats) -> None:
    """Validate meta-learning statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any statistic value is invalid.

    Examples:
        >>> stats = MetaLearningStats(
        ...     meta_train_accuracy=0.85,
        ...     meta_test_accuracy=0.75,
        ...     adaptation_steps=5.0,
        ...     generalization_gap=0.10,
        ... )
        >>> validate_meta_learning_stats(stats)

        >>> bad_stats = MetaLearningStats(
        ...     meta_train_accuracy=1.5,
        ...     meta_test_accuracy=0.75,
        ...     adaptation_steps=5.0,
        ...     generalization_gap=0.10,
        ... )
        >>> validate_meta_learning_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: meta_train_accuracy must be between 0 and 1, got 1.5
    """
    if not 0 <= stats.meta_train_accuracy <= 1:
        msg = (
            f"meta_train_accuracy must be between 0 and 1, "
            f"got {stats.meta_train_accuracy}"
        )
        raise ValueError(msg)
    if not 0 <= stats.meta_test_accuracy <= 1:
        msg = (
            f"meta_test_accuracy must be between 0 and 1, "
            f"got {stats.meta_test_accuracy}"
        )
        raise ValueError(msg)
    if stats.adaptation_steps < 0:
        msg = f"adaptation_steps must be non-negative, got {stats.adaptation_steps}"
        raise ValueError(msg)


def create_maml_config(
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    first_order: bool = False,
) -> MAMLConfig:
    """Create a MAML configuration with validation.

    Args:
        inner_lr: Learning rate for task-specific adaptation.
        outer_lr: Learning rate for meta-parameter updates.
        inner_steps: Number of gradient steps for adaptation.
        first_order: Whether to use first-order approximation.

    Returns:
        Validated MAMLConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_maml_config()
        >>> config.inner_lr
        0.01
        >>> config.inner_steps
        5

        >>> config = create_maml_config(inner_lr=0.05, inner_steps=10)
        >>> config.inner_lr
        0.05
        >>> config.inner_steps
        10

        >>> create_maml_config(inner_lr=-0.01)
        Traceback (most recent call last):
            ...
        ValueError: inner_lr must be positive, got -0.01
    """
    config = MAMLConfig(
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        first_order=first_order,
    )
    validate_maml_config(config)
    return config


def create_protonet_config(
    distance_metric: str = "euclidean",
    embedding_dim: int = 64,
    normalize: bool = True,
) -> ProtoNetConfig:
    """Create a ProtoNet configuration with validation.

    Args:
        distance_metric: Distance metric to use (euclidean, cosine).
        embedding_dim: Dimension of the embedding space.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        Validated ProtoNetConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_protonet_config()
        >>> config.distance_metric
        'euclidean'
        >>> config.embedding_dim
        64

        >>> config = create_protonet_config(distance_metric="cosine", embedding_dim=128)
        >>> config.distance_metric
        'cosine'
        >>> config.embedding_dim
        128

        >>> create_protonet_config(distance_metric="invalid")
        Traceback (most recent call last):
            ...
        ValueError: distance_metric must be 'euclidean' or 'cosine', got 'invalid'
    """
    config = ProtoNetConfig(
        distance_metric=distance_metric,
        embedding_dim=embedding_dim,
        normalize=normalize,
    )
    validate_protonet_config(config)
    return config


def create_episode_config(
    n_way: int = 5,
    k_shot: int = 1,
    query_size: int = 15,
) -> EpisodeConfig:
    """Create an episode configuration with validation.

    Args:
        n_way: Number of classes per episode.
        k_shot: Number of support examples per class.
        query_size: Number of query examples per class.

    Returns:
        Validated EpisodeConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_episode_config()
        >>> config.n_way
        5
        >>> config.k_shot
        1
        >>> config.query_size
        15

        >>> config = create_episode_config(n_way=10, k_shot=5)
        >>> config.n_way
        10
        >>> config.k_shot
        5

        >>> create_episode_config(n_way=0)
        Traceback (most recent call last):
            ...
        ValueError: n_way must be positive, got 0
    """
    config = EpisodeConfig(
        n_way=n_way,
        k_shot=k_shot,
        query_size=query_size,
    )
    validate_episode_config(config)
    return config


def create_meta_learning_config(
    method: str | MetaLearningMethod = MetaLearningMethod.MAML,
    maml_config: MAMLConfig | None = None,
    protonet_config: ProtoNetConfig | None = None,
    episode_config: EpisodeConfig | None = None,
) -> MetaLearningConfig:
    """Create a meta-learning configuration with validation.

    Args:
        method: Meta-learning method to use.
        maml_config: Configuration for MAML (if applicable).
        protonet_config: Configuration for ProtoNet (if applicable).
        episode_config: Configuration for episodic training.

    Returns:
        Validated MetaLearningConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_meta_learning_config()
        >>> config.method
        <MetaLearningMethod.MAML: 'maml'>
        >>> config.episode_config.n_way
        5

        >>> config = create_meta_learning_config(method="protonet")
        >>> config.method
        <MetaLearningMethod.PROTONET: 'protonet'>

        >>> create_meta_learning_config(method="invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if isinstance(method, str):
        method = get_meta_learning_method(method)

    if episode_config is None:
        episode_config = create_episode_config()

    # Create default configs based on method if not provided
    if method in (MetaLearningMethod.MAML, MetaLearningMethod.REPTILE):
        if maml_config is None:
            maml_config = create_maml_config()
    elif (
        method
        in (
            MetaLearningMethod.PROTONET,
            MetaLearningMethod.MATCHING_NET,
            MetaLearningMethod.RELATION_NET,
        )
        and protonet_config is None
    ):
        protonet_config = create_protonet_config()

    config = MetaLearningConfig(
        method=method,
        maml_config=maml_config,
        protonet_config=protonet_config,
        episode_config=episode_config,
    )
    validate_meta_learning_config(config)
    return config


def create_meta_learning_stats(
    meta_train_accuracy: float = 0.0,
    meta_test_accuracy: float = 0.0,
    adaptation_steps: float = 0.0,
    generalization_gap: float = 0.0,
) -> MetaLearningStats:
    """Create meta-learning statistics with validation.

    Args:
        meta_train_accuracy: Accuracy on meta-training tasks.
        meta_test_accuracy: Accuracy on meta-test tasks.
        adaptation_steps: Average steps needed for adaptation.
        generalization_gap: Gap between train and test accuracy.

    Returns:
        Validated MetaLearningStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_meta_learning_stats()
        >>> stats.meta_train_accuracy
        0.0
        >>> stats.adaptation_steps
        0.0

        >>> stats = create_meta_learning_stats(
        ...     meta_train_accuracy=0.85,
        ...     meta_test_accuracy=0.75,
        ... )
        >>> stats.meta_train_accuracy
        0.85

        >>> create_meta_learning_stats(meta_train_accuracy=1.5)
        Traceback (most recent call last):
            ...
        ValueError: meta_train_accuracy must be between 0 and 1, got 1.5
    """
    stats = MetaLearningStats(
        meta_train_accuracy=meta_train_accuracy,
        meta_test_accuracy=meta_test_accuracy,
        adaptation_steps=adaptation_steps,
        generalization_gap=generalization_gap,
    )
    validate_meta_learning_stats(stats)
    return stats


def list_meta_learning_methods() -> list[str]:
    """List all available meta-learning methods.

    Returns:
        Sorted list of meta-learning method names.

    Examples:
        >>> methods = list_meta_learning_methods()
        >>> "maml" in methods
        True
        >>> "protonet" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_META_LEARNING_METHODS)


def list_task_distributions() -> list[str]:
    """List all available task distribution strategies.

    Returns:
        Sorted list of task distribution names.

    Examples:
        >>> distributions = list_task_distributions()
        >>> "uniform" in distributions
        True
        >>> "curriculum" in distributions
        True
        >>> distributions == sorted(distributions)
        True
    """
    return sorted(VALID_TASK_DISTRIBUTIONS)


def list_adaptation_strategies() -> list[str]:
    """List all available adaptation strategies.

    Returns:
        Sorted list of adaptation strategy names.

    Examples:
        >>> strategies = list_adaptation_strategies()
        >>> "gradient" in strategies
        True
        >>> "metric" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_ADAPTATION_STRATEGIES)


def get_meta_learning_method(name: str) -> MetaLearningMethod:
    """Get meta-learning method enum from string name.

    Args:
        name: Name of the meta-learning method.

    Returns:
        Corresponding MetaLearningMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_meta_learning_method("maml")
        <MetaLearningMethod.MAML: 'maml'>
        >>> get_meta_learning_method("protonet")
        <MetaLearningMethod.PROTONET: 'protonet'>

        >>> get_meta_learning_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if name not in VALID_META_LEARNING_METHODS:
        msg = f"method must be one of {VALID_META_LEARNING_METHODS}, got '{name}'"
        raise ValueError(msg)
    return MetaLearningMethod(name)


def get_task_distribution(name: str) -> TaskDistribution:
    """Get task distribution enum from string name.

    Args:
        name: Name of the task distribution.

    Returns:
        Corresponding TaskDistribution enum.

    Raises:
        ValueError: If distribution name is invalid.

    Examples:
        >>> get_task_distribution("uniform")
        <TaskDistribution.UNIFORM: 'uniform'>
        >>> get_task_distribution("curriculum")
        <TaskDistribution.CURRICULUM: 'curriculum'>

        >>> get_task_distribution("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Invalid task_distribution ...
    """
    if name not in VALID_TASK_DISTRIBUTIONS:
        valid = VALID_TASK_DISTRIBUTIONS
        raise ValueError(f"Invalid task_distribution {name!r}, must be in {valid}")
    return TaskDistribution(name)


def get_adaptation_strategy(name: str) -> AdaptationStrategy:
    """Get adaptation strategy enum from string name.

    Args:
        name: Name of the adaptation strategy.

    Returns:
        Corresponding AdaptationStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_adaptation_strategy("gradient")
        <AdaptationStrategy.GRADIENT: 'gradient'>
        >>> get_adaptation_strategy("metric")
        <AdaptationStrategy.METRIC: 'metric'>

        >>> get_adaptation_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: adaptation_strategy must be one of ...
    """
    if name not in VALID_ADAPTATION_STRATEGIES:
        valid = VALID_ADAPTATION_STRATEGIES
        msg = f"adaptation_strategy must be one of {valid}, got '{name}'"
        raise ValueError(msg)
    return AdaptationStrategy(name)


def calculate_prototypes(
    embeddings: tuple[tuple[float, ...], ...],
    labels: tuple[int, ...],
    n_classes: int,
) -> tuple[tuple[float, ...], ...]:
    """Calculate class prototypes from support set embeddings.

    Prototypes are the mean embedding vectors for each class.

    Args:
        embeddings: Tuple of embedding vectors (one per sample).
        labels: Tuple of class labels for each embedding.
        n_classes: Number of classes.

    Returns:
        Tuple of prototype vectors (one per class).

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> embeddings = ((1.0, 0.0), (1.2, 0.1), (0.0, 1.0), (0.1, 1.1))
        >>> labels = (0, 0, 1, 1)
        >>> prototypes = calculate_prototypes(embeddings, labels, 2)
        >>> len(prototypes)
        2
        >>> prototypes[0]
        (1.1, 0.05)
        >>> prototypes[1]
        (0.05, 1.05)

        >>> calculate_prototypes((), (), 2)
        Traceback (most recent call last):
            ...
        ValueError: embeddings cannot be empty

        >>> calculate_prototypes(((1.0,),), (0, 1), 2)
        Traceback (most recent call last):
            ...
        ValueError: embeddings and labels must have same length
    """
    if not embeddings:
        msg = "embeddings cannot be empty"
        raise ValueError(msg)
    if not labels:
        msg = "labels cannot be empty"
        raise ValueError(msg)
    if len(embeddings) != len(labels):
        msg = "embeddings and labels must have same length"
        raise ValueError(msg)
    if n_classes <= 0:
        msg = f"n_classes must be positive, got {n_classes}"
        raise ValueError(msg)

    embedding_dim = len(embeddings[0])

    # Group embeddings by class
    class_embeddings: dict[int, list[tuple[float, ...]]] = {
        c: [] for c in range(n_classes)
    }
    for emb, label in zip(embeddings, labels, strict=True):
        if label < 0 or label >= n_classes:
            msg = f"label {label} out of range [0, {n_classes})"
            raise ValueError(msg)
        class_embeddings[label].append(emb)

    # Calculate mean for each class
    prototypes: list[tuple[float, ...]] = []
    for c in range(n_classes):
        if not class_embeddings[c]:
            # No samples for this class, use zeros
            prototypes.append(tuple(0.0 for _ in range(embedding_dim)))
        else:
            n_samples = len(class_embeddings[c])
            mean_emb = tuple(
                sum(emb[d] for emb in class_embeddings[c]) / n_samples
                for d in range(embedding_dim)
            )
            prototypes.append(mean_emb)

    return tuple(prototypes)


def compute_episode_accuracy(
    query_embeddings: tuple[tuple[float, ...], ...],
    query_labels: tuple[int, ...],
    prototypes: tuple[tuple[float, ...], ...],
    distance_metric: str = "euclidean",
) -> float:
    """Compute accuracy on query set using prototype classification.

    Args:
        query_embeddings: Embeddings for query samples.
        query_labels: True labels for query samples.
        prototypes: Class prototype embeddings.
        distance_metric: Distance metric to use (euclidean, cosine).

    Returns:
        Classification accuracy between 0 and 1.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> prototypes = ((1.0, 0.0), (0.0, 1.0))
        >>> query_embeddings = ((0.9, 0.1), (0.1, 0.9), (0.8, 0.2))
        >>> query_labels = (0, 1, 0)
        >>> acc = compute_episode_accuracy(query_embeddings, query_labels, prototypes)
        >>> acc == 1.0
        True

        >>> compute_episode_accuracy((), (), prototypes)
        Traceback (most recent call last):
            ...
        ValueError: query_embeddings cannot be empty
    """
    if not query_embeddings:
        msg = "query_embeddings cannot be empty"
        raise ValueError(msg)
    if not query_labels:
        msg = "query_labels cannot be empty"
        raise ValueError(msg)
    if len(query_embeddings) != len(query_labels):
        msg = "query_embeddings and query_labels must have same length"
        raise ValueError(msg)
    if not prototypes:
        msg = "prototypes cannot be empty"
        raise ValueError(msg)

    valid_metrics = {"euclidean", "cosine"}
    if distance_metric not in valid_metrics:
        msg = f"Invalid distance_metric {distance_metric!r}, use 'euclidean'/'cosine'"
        raise ValueError(msg)

    correct = 0
    for query_emb, true_label in zip(query_embeddings, query_labels, strict=True):
        # Compute distance to each prototype
        min_dist = float("inf")
        predicted_label = 0

        for proto_idx, prototype in enumerate(prototypes):
            if distance_metric == "euclidean":
                dist = math.sqrt(
                    sum((q - p) ** 2 for q, p in zip(query_emb, prototype, strict=True))
                )
            else:  # cosine
                # Cosine distance = 1 - cosine_similarity
                dot = sum(q * p for q, p in zip(query_emb, prototype, strict=True))
                norm_q = math.sqrt(sum(q**2 for q in query_emb))
                norm_p = math.sqrt(sum(p**2 for p in prototype))
                if norm_q > 0 and norm_p > 0:
                    cosine_sim = dot / (norm_q * norm_p)
                    dist = 1.0 - cosine_sim
                else:
                    dist = float("inf")

            if dist < min_dist:
                min_dist = dist
                predicted_label = proto_idx

        if predicted_label == true_label:
            correct += 1

    return correct / len(query_embeddings)


def estimate_adaptation_cost(
    method: MetaLearningMethod,
    inner_steps: int,
    n_way: int,
    k_shot: int,
    model_parameters: int,
) -> dict[str, float]:
    """Estimate computational cost of adaptation.

    Args:
        method: Meta-learning method being used.
        inner_steps: Number of inner loop gradient steps.
        n_way: Number of classes in the episode.
        k_shot: Number of support examples per class.
        model_parameters: Number of model parameters.

    Returns:
        Dictionary with cost estimates (flops, memory_gb).

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> cost = estimate_adaptation_cost(
        ...     method=MetaLearningMethod.MAML,
        ...     inner_steps=5,
        ...     n_way=5,
        ...     k_shot=1,
        ...     model_parameters=1_000_000,
        ... )
        >>> "flops" in cost
        True
        >>> "memory_gb" in cost
        True
        >>> cost["flops"] > 0
        True

        >>> estimate_adaptation_cost(
        ...     MetaLearningMethod.MAML, 0, 5, 1, 1000000
        ... )
        Traceback (most recent call last):
            ...
        ValueError: inner_steps must be positive, got 0
    """
    if inner_steps <= 0:
        msg = f"inner_steps must be positive, got {inner_steps}"
        raise ValueError(msg)
    if n_way <= 0:
        msg = f"n_way must be positive, got {n_way}"
        raise ValueError(msg)
    if k_shot <= 0:
        msg = f"k_shot must be positive, got {k_shot}"
        raise ValueError(msg)
    if model_parameters <= 0:
        msg = f"model_parameters must be positive, got {model_parameters}"
        raise ValueError(msg)

    support_samples = n_way * k_shot

    # Base FLOPs for forward pass
    base_flops = 2 * model_parameters * support_samples

    # Memory for parameters (4 bytes per float32)
    base_memory = model_parameters * 4 / (1024**3)

    if method == MetaLearningMethod.MAML:
        # MAML needs to compute second-order gradients (expensive)
        flops = base_flops * inner_steps * 3  # Forward + backward + Hessian approx
        memory = base_memory * (inner_steps + 1)  # Store intermediate parameters
    elif method == MetaLearningMethod.REPTILE:
        # Reptile is first-order, cheaper than MAML
        flops = base_flops * inner_steps * 2  # Forward + backward
        memory = base_memory * 2  # Current + initial params
    elif method in (
        MetaLearningMethod.PROTONET,
        MetaLearningMethod.MATCHING_NET,
        MetaLearningMethod.RELATION_NET,
    ):
        # Metric-based methods only need forward pass
        flops = base_flops
        memory = base_memory
    else:
        flops = base_flops * inner_steps
        memory = base_memory

    return {
        "flops": flops,
        "memory_gb": memory,
    }


def evaluate_generalization(
    train_accuracies: tuple[float, ...],
    test_accuracies: tuple[float, ...],
) -> dict[str, float]:
    """Evaluate meta-learning generalization from train/test accuracies.

    Args:
        train_accuracies: Accuracies on meta-training tasks.
        test_accuracies: Accuracies on meta-test tasks.

    Returns:
        Dictionary with generalization metrics.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> train_acc = (0.90, 0.85, 0.88)
        >>> test_acc = (0.80, 0.75, 0.78)
        >>> metrics = evaluate_generalization(train_acc, test_acc)
        >>> "mean_train_accuracy" in metrics
        True
        >>> "mean_test_accuracy" in metrics
        True
        >>> "generalization_gap" in metrics
        True
        >>> metrics["generalization_gap"] > 0
        True

        >>> evaluate_generalization((), (0.8,))
        Traceback (most recent call last):
            ...
        ValueError: train_accuracies cannot be empty
    """
    if not train_accuracies:
        msg = "train_accuracies cannot be empty"
        raise ValueError(msg)
    if not test_accuracies:
        msg = "test_accuracies cannot be empty"
        raise ValueError(msg)

    mean_train = sum(train_accuracies) / len(train_accuracies)
    mean_test = sum(test_accuracies) / len(test_accuracies)

    # Generalization gap
    gap = mean_train - mean_test

    # Standard deviations
    train_std = math.sqrt(
        sum((a - mean_train) ** 2 for a in train_accuracies) / len(train_accuracies)
    )
    test_std = math.sqrt(
        sum((a - mean_test) ** 2 for a in test_accuracies) / len(test_accuracies)
    )

    return {
        "mean_train_accuracy": mean_train,
        "mean_test_accuracy": mean_test,
        "generalization_gap": gap,
        "train_std": train_std,
        "test_std": test_std,
    }


def format_meta_learning_stats(stats: MetaLearningStats) -> str:
    """Format meta-learning statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_meta_learning_stats(
        ...     meta_train_accuracy=0.85,
        ...     meta_test_accuracy=0.75,
        ...     adaptation_steps=5.0,
        ...     generalization_gap=0.10,
        ... )
        >>> formatted = format_meta_learning_stats(stats)
        >>> "Meta-Train Accuracy: 85.0%" in formatted
        True
        >>> "Meta-Test Accuracy: 75.0%" in formatted
        True
        >>> "Generalization Gap: 10.0%" in formatted
        True
    """
    return (
        f"Meta-Learning Stats:\n"
        f"  Meta-Train Accuracy: {stats.meta_train_accuracy * 100:.1f}%\n"
        f"  Meta-Test Accuracy: {stats.meta_test_accuracy * 100:.1f}%\n"
        f"  Adaptation Steps: {stats.adaptation_steps:.1f}\n"
        f"  Generalization Gap: {stats.generalization_gap * 100:.1f}%"
    )


def get_recommended_meta_learning_config(
    task_type: str,
    available_samples_per_class: int,
) -> MetaLearningConfig:
    """Get recommended meta-learning configuration for a scenario.

    Args:
        task_type: Type of task (classification, regression, reinforcement).
        available_samples_per_class: Number of samples available per class.

    Returns:
        Recommended MetaLearningConfig for the given scenario.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> config = get_recommended_meta_learning_config("classification", 5)
        >>> config.method
        <MetaLearningMethod.PROTONET: 'protonet'>

        >>> config = get_recommended_meta_learning_config("classification", 20)
        >>> config.method
        <MetaLearningMethod.MAML: 'maml'>

        >>> config = get_recommended_meta_learning_config("regression", 10)
        >>> config.method
        <MetaLearningMethod.REPTILE: 'reptile'>

        >>> get_recommended_meta_learning_config("unknown", 10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task_type must be one of ...
    """
    valid_tasks = frozenset({"classification", "regression", "reinforcement"})
    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)
    if available_samples_per_class <= 0:
        msg = (
            f"available_samples_per_class must be positive, "
            f"got {available_samples_per_class}"
        )
        raise ValueError(msg)

    if task_type == "classification":
        if available_samples_per_class <= 5:
            # Few samples: metric-based methods work better
            return create_meta_learning_config(
                method=MetaLearningMethod.PROTONET,
                protonet_config=create_protonet_config(
                    distance_metric="euclidean",
                    embedding_dim=64,
                    normalize=True,
                ),
                episode_config=create_episode_config(
                    n_way=5,
                    k_shot=min(available_samples_per_class, 1),
                    query_size=15,
                ),
            )
        else:
            # More samples: gradient-based methods can be used
            return create_meta_learning_config(
                method=MetaLearningMethod.MAML,
                maml_config=create_maml_config(
                    inner_lr=0.01,
                    outer_lr=0.001,
                    inner_steps=5,
                    first_order=False,
                ),
                episode_config=create_episode_config(
                    n_way=5,
                    k_shot=min(available_samples_per_class // 2, 5),
                    query_size=15,
                ),
            )
    elif task_type == "regression":
        # Reptile is good for regression tasks
        return create_meta_learning_config(
            method=MetaLearningMethod.REPTILE,
            maml_config=create_maml_config(
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=10,
                first_order=True,
            ),
            episode_config=create_episode_config(
                n_way=1,  # Regression is typically single-task
                k_shot=min(available_samples_per_class, 10),
                query_size=min(available_samples_per_class, 10),
            ),
        )
    else:  # reinforcement
        # MAML with first-order for RL
        return create_meta_learning_config(
            method=MetaLearningMethod.MAML,
            maml_config=create_maml_config(
                inner_lr=0.1,
                outer_lr=0.001,
                inner_steps=1,
                first_order=True,
            ),
            episode_config=create_episode_config(
                n_way=1,
                k_shot=min(available_samples_per_class, 20),
                query_size=min(available_samples_per_class, 20),
            ),
        )
