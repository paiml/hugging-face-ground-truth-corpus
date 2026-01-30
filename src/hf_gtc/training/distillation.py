"""Knowledge Distillation training utilities.

This module provides functions for configuring and running knowledge
distillation training, enabling model compression by transferring
knowledge from larger teacher models to smaller student models.

Examples:
    >>> from hf_gtc.training.distillation import (
    ...     create_distillation_config,
    ...     DistillationMethod,
    ... )
    >>> config = create_distillation_config(temperature=4.0, alpha=0.7)
    >>> config.temperature
    4.0
    >>> config.alpha
    0.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DistillationMethod(Enum):
    """Supported knowledge distillation methods.

    Attributes:
        RESPONSE_BASED: Distill from soft labels (logits).
        FEATURE_BASED: Match intermediate feature representations.
        ATTENTION_TRANSFER: Transfer attention maps from teacher.
        SELF_DISTILLATION: Model distills knowledge to itself.
        PROGRESSIVE: Gradually increase distillation difficulty.

    Examples:
        >>> DistillationMethod.RESPONSE_BASED.value
        'response_based'
        >>> DistillationMethod.FEATURE_BASED.value
        'feature_based'
        >>> DistillationMethod.ATTENTION_TRANSFER.value
        'attention_transfer'
    """

    RESPONSE_BASED = "response_based"
    FEATURE_BASED = "feature_based"
    ATTENTION_TRANSFER = "attention_transfer"
    SELF_DISTILLATION = "self_distillation"
    PROGRESSIVE = "progressive"


class DistillationLoss(Enum):
    """Loss functions for knowledge distillation.

    Attributes:
        KL_DIVERGENCE: Kullback-Leibler divergence for soft labels.
        MSE: Mean squared error for feature matching.
        COSINE: Cosine similarity loss for embeddings.
        CROSS_ENTROPY: Cross-entropy with soft targets.
        COMBINED: Weighted combination of hard and soft losses.

    Examples:
        >>> DistillationLoss.KL_DIVERGENCE.value
        'kl_divergence'
        >>> DistillationLoss.MSE.value
        'mse'
        >>> DistillationLoss.COMBINED.value
        'combined'
    """

    KL_DIVERGENCE = "kl_divergence"
    MSE = "mse"
    COSINE = "cosine"
    CROSS_ENTROPY = "cross_entropy"
    COMBINED = "combined"


class TemperatureSchedule(Enum):
    """Temperature scheduling strategies during distillation.

    Attributes:
        CONSTANT: Fixed temperature throughout training.
        LINEAR_DECAY: Linearly decrease temperature over time.
        COSINE_DECAY: Cosine annealing of temperature.
        EXPONENTIAL_DECAY: Exponentially decrease temperature.
        WARMUP: Warm up temperature then hold constant.

    Examples:
        >>> TemperatureSchedule.CONSTANT.value
        'constant'
        >>> TemperatureSchedule.LINEAR_DECAY.value
        'linear_decay'
        >>> TemperatureSchedule.COSINE_DECAY.value
        'cosine_decay'
    """

    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    COSINE_DECAY = "cosine_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    WARMUP = "warmup"


class StudentInitialization(Enum):
    """Initialization strategies for student model.

    Attributes:
        RANDOM: Random initialization.
        TEACHER_SUBSET: Initialize from subset of teacher layers.
        PRETRAINED: Use pretrained weights.
        PRUNED: Initialize from pruned teacher.

    Examples:
        >>> StudentInitialization.RANDOM.value
        'random'
        >>> StudentInitialization.TEACHER_SUBSET.value
        'teacher_subset'
        >>> StudentInitialization.PRETRAINED.value
        'pretrained'
    """

    RANDOM = "random"
    TEACHER_SUBSET = "teacher_subset"
    PRETRAINED = "pretrained"
    PRUNED = "pruned"


VALID_DISTILLATION_METHODS = frozenset(m.value for m in DistillationMethod)
VALID_DISTILLATION_LOSSES = frozenset(loss.value for loss in DistillationLoss)
VALID_TEMPERATURE_SCHEDULES = frozenset(s.value for s in TemperatureSchedule)
VALID_STUDENT_INITIALIZATIONS = frozenset(i.value for i in StudentInitialization)


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    """Configuration for teacher model in distillation.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path.
        num_layers: Number of layers in teacher model.
        hidden_size: Hidden dimension of teacher model.
        output_hidden_states: Whether to output intermediate hidden states.
        output_attentions: Whether to output attention weights.

    Examples:
        >>> config = TeacherConfig(
        ...     model_name_or_path="bert-large-uncased",
        ...     num_layers=24,
        ...     hidden_size=1024,
        ...     output_hidden_states=True,
        ...     output_attentions=False,
        ... )
        >>> config.num_layers
        24
        >>> config.hidden_size
        1024
    """

    model_name_or_path: str
    num_layers: int
    hidden_size: int
    output_hidden_states: bool
    output_attentions: bool


@dataclass(frozen=True, slots=True)
class StudentConfig:
    """Configuration for student model in distillation.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path.
        num_layers: Number of layers in student model.
        hidden_size: Hidden dimension of student model.
        initialization: How to initialize student weights.
        layer_mapping: Mapping from student to teacher layers.

    Examples:
        >>> config = StudentConfig(
        ...     model_name_or_path="distilbert-base-uncased",
        ...     num_layers=6,
        ...     hidden_size=768,
        ...     initialization=StudentInitialization.PRETRAINED,
        ...     layer_mapping=(0, 4, 8, 12, 16, 20),
        ... )
        >>> config.num_layers
        6
        >>> config.initialization
        <StudentInitialization.PRETRAINED: 'pretrained'>
    """

    model_name_or_path: str
    num_layers: int
    hidden_size: int
    initialization: StudentInitialization
    layer_mapping: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DistillationLossConfig:
    """Configuration for distillation loss computation.

    Attributes:
        loss_type: Type of distillation loss to use.
        temperature: Softmax temperature for soft labels.
        alpha: Weight for distillation loss vs task loss.
        beta: Weight for feature matching loss.
        normalize_features: Whether to normalize features before matching.

    Examples:
        >>> config = DistillationLossConfig(
        ...     loss_type=DistillationLoss.KL_DIVERGENCE,
        ...     temperature=4.0,
        ...     alpha=0.7,
        ...     beta=0.0,
        ...     normalize_features=True,
        ... )
        >>> config.temperature
        4.0
        >>> config.alpha
        0.7
    """

    loss_type: DistillationLoss
    temperature: float
    alpha: float
    beta: float
    normalize_features: bool


@dataclass(frozen=True, slots=True)
class FeatureMatchingConfig:
    """Configuration for feature-based distillation.

    Attributes:
        match_hidden_states: Whether to match hidden state representations.
        match_attention: Whether to match attention patterns.
        hidden_layer_indices: Which teacher layers to match.
        attention_layer_indices: Which attention layers to match.
        projection_dim: Dimension for projection if sizes differ.

    Examples:
        >>> config = FeatureMatchingConfig(
        ...     match_hidden_states=True,
        ...     match_attention=False,
        ...     hidden_layer_indices=(4, 8, 12),
        ...     attention_layer_indices=(),
        ...     projection_dim=768,
        ... )
        >>> config.match_hidden_states
        True
        >>> len(config.hidden_layer_indices)
        3
    """

    match_hidden_states: bool
    match_attention: bool
    hidden_layer_indices: tuple[int, ...]
    attention_layer_indices: tuple[int, ...]
    projection_dim: int


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Main configuration for knowledge distillation training.

    Attributes:
        method: Distillation method to use.
        temperature: Softmax temperature for soft labels.
        alpha: Weight for distillation loss (1-alpha for task loss).
        temperature_schedule: How to adjust temperature during training.
        final_temperature: Final temperature for scheduled decay.
        warmup_steps: Number of warmup steps for temperature.

    Examples:
        >>> config = DistillationConfig(
        ...     method=DistillationMethod.RESPONSE_BASED,
        ...     temperature=4.0,
        ...     alpha=0.7,
        ...     temperature_schedule=TemperatureSchedule.CONSTANT,
        ...     final_temperature=1.0,
        ...     warmup_steps=0,
        ... )
        >>> config.temperature
        4.0
        >>> config.method
        <DistillationMethod.RESPONSE_BASED: 'response_based'>
    """

    method: DistillationMethod
    temperature: float
    alpha: float
    temperature_schedule: TemperatureSchedule
    final_temperature: float
    warmup_steps: int


@dataclass(frozen=True, slots=True)
class DistillationStats:
    """Statistics from distillation training.

    Attributes:
        total_steps: Total training steps completed.
        distillation_loss: Average distillation loss.
        task_loss: Average task-specific loss.
        combined_loss: Average combined loss.
        temperature: Current temperature value.
        compression_ratio: Size ratio of student to teacher.

    Examples:
        >>> stats = DistillationStats(
        ...     total_steps=1000,
        ...     distillation_loss=0.5,
        ...     task_loss=0.3,
        ...     combined_loss=0.44,
        ...     temperature=4.0,
        ...     compression_ratio=0.25,
        ... )
        >>> stats.total_steps
        1000
        >>> stats.compression_ratio
        0.25
    """

    total_steps: int
    distillation_loss: float
    task_loss: float
    combined_loss: float
    temperature: float
    compression_ratio: float


def validate_distillation_config(config: DistillationConfig) -> None:
    """Validate distillation configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = DistillationConfig(
        ...     method=DistillationMethod.RESPONSE_BASED,
        ...     temperature=4.0,
        ...     alpha=0.7,
        ...     temperature_schedule=TemperatureSchedule.CONSTANT,
        ...     final_temperature=1.0,
        ...     warmup_steps=0,
        ... )
        >>> validate_distillation_config(config)

        >>> bad_config = DistillationConfig(
        ...     method=DistillationMethod.RESPONSE_BASED,
        ...     temperature=0.0,
        ...     alpha=0.7,
        ...     temperature_schedule=TemperatureSchedule.CONSTANT,
        ...     final_temperature=1.0,
        ...     warmup_steps=0,
        ... )
        >>> validate_distillation_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: temperature must be positive, got 0.0
    """
    if config.temperature <= 0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)
    if not 0 <= config.alpha <= 1:
        msg = f"alpha must be between 0 and 1, got {config.alpha}"
        raise ValueError(msg)
    if config.final_temperature <= 0:
        msg = f"final_temperature must be positive, got {config.final_temperature}"
        raise ValueError(msg)
    if config.warmup_steps < 0:
        msg = f"warmup_steps must be non-negative, got {config.warmup_steps}"
        raise ValueError(msg)


def validate_teacher_config(config: TeacherConfig) -> None:
    """Validate teacher model configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = TeacherConfig(
        ...     model_name_or_path="bert-large-uncased",
        ...     num_layers=24,
        ...     hidden_size=1024,
        ...     output_hidden_states=True,
        ...     output_attentions=False,
        ... )
        >>> validate_teacher_config(config)

        >>> bad_config = TeacherConfig(
        ...     model_name_or_path="",
        ...     num_layers=24,
        ...     hidden_size=1024,
        ...     output_hidden_states=True,
        ...     output_attentions=False,
        ... )
        >>> validate_teacher_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: model_name_or_path cannot be empty
    """
    if not config.model_name_or_path or not config.model_name_or_path.strip():
        msg = "model_name_or_path cannot be empty"
        raise ValueError(msg)
    if config.num_layers <= 0:
        msg = f"num_layers must be positive, got {config.num_layers}"
        raise ValueError(msg)
    if config.hidden_size <= 0:
        msg = f"hidden_size must be positive, got {config.hidden_size}"
        raise ValueError(msg)


def validate_student_config(config: StudentConfig) -> None:
    """Validate student model configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = StudentConfig(
        ...     model_name_or_path="distilbert-base-uncased",
        ...     num_layers=6,
        ...     hidden_size=768,
        ...     initialization=StudentInitialization.PRETRAINED,
        ...     layer_mapping=(0, 4, 8, 12, 16, 20),
        ... )
        >>> validate_student_config(config)

        >>> bad_config = StudentConfig(
        ...     model_name_or_path="distilbert-base-uncased",
        ...     num_layers=6,
        ...     hidden_size=768,
        ...     initialization=StudentInitialization.PRETRAINED,
        ...     layer_mapping=(0, 4, 8),
        ... )
        >>> validate_student_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: layer_mapping length (3) must match num_layers (6)
    """
    if not config.model_name_or_path or not config.model_name_or_path.strip():
        msg = "model_name_or_path cannot be empty"
        raise ValueError(msg)
    if config.num_layers <= 0:
        msg = f"num_layers must be positive, got {config.num_layers}"
        raise ValueError(msg)
    if config.hidden_size <= 0:
        msg = f"hidden_size must be positive, got {config.hidden_size}"
        raise ValueError(msg)
    if len(config.layer_mapping) != config.num_layers:
        msg = (
            f"layer_mapping length ({len(config.layer_mapping)}) "
            f"must match num_layers ({config.num_layers})"
        )
        raise ValueError(msg)


def validate_loss_config(config: DistillationLossConfig) -> None:
    """Validate distillation loss configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = DistillationLossConfig(
        ...     loss_type=DistillationLoss.KL_DIVERGENCE,
        ...     temperature=4.0,
        ...     alpha=0.7,
        ...     beta=0.0,
        ...     normalize_features=True,
        ... )
        >>> validate_loss_config(config)

        >>> bad_config = DistillationLossConfig(
        ...     loss_type=DistillationLoss.KL_DIVERGENCE,
        ...     temperature=-1.0,
        ...     alpha=0.7,
        ...     beta=0.0,
        ...     normalize_features=True,
        ... )
        >>> validate_loss_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: temperature must be positive, got -1.0
    """
    if config.temperature <= 0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)
    if not 0 <= config.alpha <= 1:
        msg = f"alpha must be between 0 and 1, got {config.alpha}"
        raise ValueError(msg)
    if config.beta < 0:
        msg = f"beta must be non-negative, got {config.beta}"
        raise ValueError(msg)


def validate_feature_matching_config(config: FeatureMatchingConfig) -> None:
    """Validate feature matching configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = FeatureMatchingConfig(
        ...     match_hidden_states=True,
        ...     match_attention=False,
        ...     hidden_layer_indices=(4, 8, 12),
        ...     attention_layer_indices=(),
        ...     projection_dim=768,
        ... )
        >>> validate_feature_matching_config(config)

        >>> bad_config = FeatureMatchingConfig(
        ...     match_hidden_states=True,
        ...     match_attention=False,
        ...     hidden_layer_indices=(4, 8, 12),
        ...     attention_layer_indices=(),
        ...     projection_dim=0,
        ... )
        >>> validate_feature_matching_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: projection_dim must be positive, got 0
    """
    if config.projection_dim <= 0:
        msg = f"projection_dim must be positive, got {config.projection_dim}"
        raise ValueError(msg)
    if config.match_hidden_states and not config.hidden_layer_indices:
        msg = "hidden_layer_indices required when match_hidden_states is True"
        raise ValueError(msg)
    if config.match_attention and not config.attention_layer_indices:
        msg = "attention_layer_indices required when match_attention is True"
        raise ValueError(msg)


def create_distillation_config(
    method: str | DistillationMethod = DistillationMethod.RESPONSE_BASED,
    temperature: float = 4.0,
    alpha: float = 0.5,
    temperature_schedule: str | TemperatureSchedule = TemperatureSchedule.CONSTANT,
    final_temperature: float = 1.0,
    warmup_steps: int = 0,
) -> DistillationConfig:
    """Create a distillation configuration with validation.

    Args:
        method: Distillation method to use.
        temperature: Softmax temperature for soft labels.
        alpha: Weight for distillation loss.
        temperature_schedule: How to adjust temperature.
        final_temperature: Final temperature for decay schedules.
        warmup_steps: Number of warmup steps.

    Returns:
        Validated DistillationConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_distillation_config()
        >>> config.temperature
        4.0
        >>> config.alpha
        0.5

        >>> config = create_distillation_config(method="feature_based", alpha=0.8)
        >>> config.method
        <DistillationMethod.FEATURE_BASED: 'feature_based'>
        >>> config.alpha
        0.8

        >>> create_distillation_config(temperature=0)
        Traceback (most recent call last):
            ...
        ValueError: temperature must be positive, got 0
    """
    if isinstance(method, str):
        method = get_distillation_method(method)
    if isinstance(temperature_schedule, str):
        temperature_schedule = get_temperature_schedule(temperature_schedule)

    config = DistillationConfig(
        method=method,
        temperature=temperature,
        alpha=alpha,
        temperature_schedule=temperature_schedule,
        final_temperature=final_temperature,
        warmup_steps=warmup_steps,
    )
    validate_distillation_config(config)
    return config


def create_teacher_config(
    model_name_or_path: str,
    num_layers: int = 12,
    hidden_size: int = 768,
    output_hidden_states: bool = True,
    output_attentions: bool = False,
) -> TeacherConfig:
    """Create a teacher model configuration with validation.

    Args:
        model_name_or_path: HuggingFace model identifier.
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        output_hidden_states: Whether to output hidden states.
        output_attentions: Whether to output attention weights.

    Returns:
        Validated TeacherConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_teacher_config("bert-base-uncased")
        >>> config.num_layers
        12
        >>> config.hidden_size
        768

        >>> config = create_teacher_config(
        ...     "bert-large-uncased",
        ...     num_layers=24,
        ...     hidden_size=1024,
        ... )
        >>> config.num_layers
        24

        >>> create_teacher_config("")
        Traceback (most recent call last):
            ...
        ValueError: model_name_or_path cannot be empty
    """
    config = TeacherConfig(
        model_name_or_path=model_name_or_path,
        num_layers=num_layers,
        hidden_size=hidden_size,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    validate_teacher_config(config)
    return config


def create_student_config(
    model_name_or_path: str,
    num_layers: int = 6,
    hidden_size: int = 768,
    initialization: str | StudentInitialization = StudentInitialization.PRETRAINED,
    layer_mapping: tuple[int, ...] | None = None,
) -> StudentConfig:
    """Create a student model configuration with validation.

    Args:
        model_name_or_path: HuggingFace model identifier.
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        initialization: Weight initialization strategy.
        layer_mapping: Mapping from student to teacher layers.

    Returns:
        Validated StudentConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_student_config("distilbert-base-uncased")
        >>> config.num_layers
        6
        >>> len(config.layer_mapping)
        6

        >>> config = create_student_config(
        ...     "my-student-model",
        ...     num_layers=4,
        ...     layer_mapping=(0, 4, 8, 11),
        ... )
        >>> config.layer_mapping
        (0, 4, 8, 11)

        >>> create_student_config("")
        Traceback (most recent call last):
            ...
        ValueError: model_name_or_path cannot be empty
    """
    if isinstance(initialization, str):
        initialization = get_student_initialization(initialization)

    if layer_mapping is None:
        layer_mapping = tuple(range(num_layers))

    config = StudentConfig(
        model_name_or_path=model_name_or_path,
        num_layers=num_layers,
        hidden_size=hidden_size,
        initialization=initialization,
        layer_mapping=layer_mapping,
    )
    validate_student_config(config)
    return config


def create_loss_config(
    loss_type: str | DistillationLoss = DistillationLoss.KL_DIVERGENCE,
    temperature: float = 4.0,
    alpha: float = 0.5,
    beta: float = 0.0,
    normalize_features: bool = True,
) -> DistillationLossConfig:
    """Create a distillation loss configuration with validation.

    Args:
        loss_type: Type of distillation loss.
        temperature: Softmax temperature.
        alpha: Weight for distillation loss.
        beta: Weight for feature matching loss.
        normalize_features: Whether to normalize features.

    Returns:
        Validated DistillationLossConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_loss_config()
        >>> config.temperature
        4.0
        >>> config.loss_type
        <DistillationLoss.KL_DIVERGENCE: 'kl_divergence'>

        >>> config = create_loss_config(loss_type="mse", beta=0.5)
        >>> config.loss_type
        <DistillationLoss.MSE: 'mse'>
        >>> config.beta
        0.5

        >>> create_loss_config(alpha=1.5)
        Traceback (most recent call last):
            ...
        ValueError: alpha must be between 0 and 1, got 1.5
    """
    if isinstance(loss_type, str):
        loss_type = get_distillation_loss(loss_type)

    config = DistillationLossConfig(
        loss_type=loss_type,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        normalize_features=normalize_features,
    )
    validate_loss_config(config)
    return config


def create_feature_matching_config(
    match_hidden_states: bool = True,
    match_attention: bool = False,
    hidden_layer_indices: tuple[int, ...] = (4, 8, 12),
    attention_layer_indices: tuple[int, ...] = (),
    projection_dim: int = 768,
) -> FeatureMatchingConfig:
    """Create a feature matching configuration with validation.

    Args:
        match_hidden_states: Whether to match hidden states.
        match_attention: Whether to match attention.
        hidden_layer_indices: Which hidden layers to match.
        attention_layer_indices: Which attention layers to match.
        projection_dim: Dimension for projection layers.

    Returns:
        Validated FeatureMatchingConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_feature_matching_config()
        >>> config.match_hidden_states
        True
        >>> config.hidden_layer_indices
        (4, 8, 12)

        >>> config = create_feature_matching_config(
        ...     match_attention=True,
        ...     attention_layer_indices=(6, 12),
        ... )
        >>> config.match_attention
        True

        >>> create_feature_matching_config(projection_dim=0)
        Traceback (most recent call last):
            ...
        ValueError: projection_dim must be positive, got 0
    """
    config = FeatureMatchingConfig(
        match_hidden_states=match_hidden_states,
        match_attention=match_attention,
        hidden_layer_indices=hidden_layer_indices,
        attention_layer_indices=attention_layer_indices,
        projection_dim=projection_dim,
    )
    validate_feature_matching_config(config)
    return config


def create_distillation_stats(
    total_steps: int = 0,
    distillation_loss: float = 0.0,
    task_loss: float = 0.0,
    combined_loss: float = 0.0,
    temperature: float = 4.0,
    compression_ratio: float = 1.0,
) -> DistillationStats:
    """Create distillation statistics.

    Args:
        total_steps: Total training steps completed.
        distillation_loss: Average distillation loss.
        task_loss: Average task-specific loss.
        combined_loss: Average combined loss.
        temperature: Current temperature value.
        compression_ratio: Size ratio of student to teacher.

    Returns:
        DistillationStats instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_distillation_stats()
        >>> stats.total_steps
        0
        >>> stats.compression_ratio
        1.0

        >>> stats = create_distillation_stats(
        ...     total_steps=1000,
        ...     distillation_loss=0.5,
        ...     compression_ratio=0.25,
        ... )
        >>> stats.total_steps
        1000

        >>> create_distillation_stats(total_steps=-1)
        Traceback (most recent call last):
            ...
        ValueError: total_steps must be non-negative, got -1
    """
    if total_steps < 0:
        msg = f"total_steps must be non-negative, got {total_steps}"
        raise ValueError(msg)
    if compression_ratio <= 0:
        msg = f"compression_ratio must be positive, got {compression_ratio}"
        raise ValueError(msg)

    return DistillationStats(
        total_steps=total_steps,
        distillation_loss=distillation_loss,
        task_loss=task_loss,
        combined_loss=combined_loss,
        temperature=temperature,
        compression_ratio=compression_ratio,
    )


def list_distillation_methods() -> list[str]:
    """List all available distillation methods.

    Returns:
        Sorted list of distillation method names.

    Examples:
        >>> methods = list_distillation_methods()
        >>> "response_based" in methods
        True
        >>> "feature_based" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_DISTILLATION_METHODS)


def list_distillation_losses() -> list[str]:
    """List all available distillation loss types.

    Returns:
        Sorted list of distillation loss names.

    Examples:
        >>> losses = list_distillation_losses()
        >>> "kl_divergence" in losses
        True
        >>> "mse" in losses
        True
        >>> losses == sorted(losses)
        True
    """
    return sorted(VALID_DISTILLATION_LOSSES)


def list_temperature_schedules() -> list[str]:
    """List all available temperature schedules.

    Returns:
        Sorted list of temperature schedule names.

    Examples:
        >>> schedules = list_temperature_schedules()
        >>> "constant" in schedules
        True
        >>> "linear_decay" in schedules
        True
        >>> schedules == sorted(schedules)
        True
    """
    return sorted(VALID_TEMPERATURE_SCHEDULES)


def list_student_initializations() -> list[str]:
    """List all available student initialization strategies.

    Returns:
        Sorted list of initialization strategy names.

    Examples:
        >>> inits = list_student_initializations()
        >>> "pretrained" in inits
        True
        >>> "random" in inits
        True
        >>> inits == sorted(inits)
        True
    """
    return sorted(VALID_STUDENT_INITIALIZATIONS)


def get_distillation_method(name: str) -> DistillationMethod:
    """Get distillation method enum from string name.

    Args:
        name: Name of the distillation method.

    Returns:
        Corresponding DistillationMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_distillation_method("response_based")
        <DistillationMethod.RESPONSE_BASED: 'response_based'>
        >>> get_distillation_method("feature_based")
        <DistillationMethod.FEATURE_BASED: 'feature_based'>

        >>> get_distillation_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if name not in VALID_DISTILLATION_METHODS:
        msg = f"method must be one of {VALID_DISTILLATION_METHODS}, got '{name}'"
        raise ValueError(msg)
    return DistillationMethod(name)


def get_distillation_loss(name: str) -> DistillationLoss:
    """Get distillation loss enum from string name.

    Args:
        name: Name of the distillation loss.

    Returns:
        Corresponding DistillationLoss enum.

    Raises:
        ValueError: If loss name is invalid.

    Examples:
        >>> get_distillation_loss("kl_divergence")
        <DistillationLoss.KL_DIVERGENCE: 'kl_divergence'>
        >>> get_distillation_loss("mse")
        <DistillationLoss.MSE: 'mse'>

        >>> get_distillation_loss("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: loss_type must be one of ...
    """
    if name not in VALID_DISTILLATION_LOSSES:
        msg = f"loss_type must be one of {VALID_DISTILLATION_LOSSES}, got '{name}'"
        raise ValueError(msg)
    return DistillationLoss(name)


def get_temperature_schedule(name: str) -> TemperatureSchedule:
    """Get temperature schedule enum from string name.

    Args:
        name: Name of the temperature schedule.

    Returns:
        Corresponding TemperatureSchedule enum.

    Raises:
        ValueError: If schedule name is invalid.

    Examples:
        >>> get_temperature_schedule("constant")
        <TemperatureSchedule.CONSTANT: 'constant'>
        >>> get_temperature_schedule("linear_decay")
        <TemperatureSchedule.LINEAR_DECAY: 'linear_decay'>

        >>> get_temperature_schedule("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: schedule must be one of ...
    """
    if name not in VALID_TEMPERATURE_SCHEDULES:
        msg = f"schedule must be one of {VALID_TEMPERATURE_SCHEDULES}, got '{name}'"
        raise ValueError(msg)
    return TemperatureSchedule(name)


def get_student_initialization(name: str) -> StudentInitialization:
    """Get student initialization enum from string name.

    Args:
        name: Name of the initialization strategy.

    Returns:
        Corresponding StudentInitialization enum.

    Raises:
        ValueError: If initialization name is invalid.

    Examples:
        >>> get_student_initialization("pretrained")
        <StudentInitialization.PRETRAINED: 'pretrained'>
        >>> get_student_initialization("random")
        <StudentInitialization.RANDOM: 'random'>

        >>> get_student_initialization("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: initialization must be one of ...
    """
    if name not in VALID_STUDENT_INITIALIZATIONS:
        msg = (
            f"initialization must be one of {VALID_STUDENT_INITIALIZATIONS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return StudentInitialization(name)


def calculate_temperature_at_step(
    config: DistillationConfig,
    current_step: int,
    total_steps: int,
) -> float:
    """Calculate temperature value at a given training step.

    Args:
        config: Distillation configuration.
        current_step: Current training step.
        total_steps: Total number of training steps.

    Returns:
        Temperature value at the current step.

    Raises:
        ValueError: If step parameters are invalid.

    Examples:
        >>> config = create_distillation_config(
        ...     temperature=4.0,
        ...     temperature_schedule="constant",
        ... )
        >>> calculate_temperature_at_step(config, 500, 1000)
        4.0

        >>> config = create_distillation_config(
        ...     temperature=4.0,
        ...     final_temperature=1.0,
        ...     temperature_schedule="linear_decay",
        ... )
        >>> round(calculate_temperature_at_step(config, 500, 1000), 1)
        2.5

        >>> calculate_temperature_at_step(config, -1, 1000)
        Traceback (most recent call last):
            ...
        ValueError: current_step must be non-negative, got -1
    """
    if current_step < 0:
        msg = f"current_step must be non-negative, got {current_step}"
        raise ValueError(msg)
    if total_steps <= 0:
        msg = f"total_steps must be positive, got {total_steps}"
        raise ValueError(msg)
    if current_step > total_steps:
        msg = f"current_step ({current_step}) cannot exceed total_steps ({total_steps})"
        raise ValueError(msg)

    schedule = config.temperature_schedule
    t_start = config.temperature
    t_end = config.final_temperature
    warmup = config.warmup_steps

    if schedule == TemperatureSchedule.CONSTANT:
        return t_start

    if current_step < warmup:
        progress = current_step / warmup if warmup > 0 else 1.0
        return t_end + (t_start - t_end) * progress

    effective_step = current_step - warmup
    effective_total = total_steps - warmup

    if effective_total <= 0:
        return t_end

    progress = effective_step / effective_total

    if schedule == TemperatureSchedule.LINEAR_DECAY:
        return t_start - (t_start - t_end) * progress
    elif schedule == TemperatureSchedule.COSINE_DECAY:
        return t_end + (t_start - t_end) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == TemperatureSchedule.EXPONENTIAL_DECAY:
        decay_rate = math.log(t_end / t_start) if t_start > 0 else 0
        return t_start * math.exp(decay_rate * progress)
    elif schedule == TemperatureSchedule.WARMUP:
        return t_start

    return t_start


def calculate_soft_labels_loss(
    student_logits: tuple[float, ...],
    teacher_logits: tuple[float, ...],
    temperature: float,
) -> float:
    """Calculate KL divergence loss between soft labels.

    This computes the distillation loss using softmax-temperature
    softened probability distributions.

    Args:
        student_logits: Student model output logits.
        teacher_logits: Teacher model output logits.
        temperature: Softmax temperature for softening.

    Returns:
        KL divergence loss value.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> student = (1.0, 2.0, 3.0)
        >>> teacher = (1.5, 2.5, 3.5)
        >>> loss = calculate_soft_labels_loss(student, teacher, 4.0)
        >>> 0 <= loss < 1
        True

        >>> calculate_soft_labels_loss((), (), 4.0)
        Traceback (most recent call last):
            ...
        ValueError: logits cannot be empty

        >>> calculate_soft_labels_loss((1.0,), (1.0, 2.0), 4.0)
        Traceback (most recent call last):
            ...
        ValueError: student and teacher logits must have same length
    """
    if not student_logits or not teacher_logits:
        msg = "logits cannot be empty"
        raise ValueError(msg)
    if len(student_logits) != len(teacher_logits):
        msg = "student and teacher logits must have same length"
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)

    def softmax_with_temperature(
        logits: tuple[float, ...], temp: float
    ) -> tuple[float, ...]:
        scaled = tuple(x / temp for x in logits)
        max_val = max(scaled)
        exp_vals = tuple(math.exp(x - max_val) for x in scaled)
        sum_exp = sum(exp_vals)
        return tuple(x / sum_exp for x in exp_vals)

    student_probs = softmax_with_temperature(student_logits, temperature)
    teacher_probs = softmax_with_temperature(teacher_logits, temperature)

    kl_div = sum(
        t * math.log(t / s) if t > 0 and s > 0 else 0
        for s, t in zip(student_probs, teacher_probs, strict=True)
    )

    return kl_div * (temperature**2)


def calculate_combined_loss(
    distillation_loss: float,
    task_loss: float,
    alpha: float,
) -> float:
    """Calculate combined distillation and task loss.

    The combined loss is: alpha * distillation_loss + (1 - alpha) * task_loss

    Args:
        distillation_loss: Loss from knowledge distillation.
        task_loss: Loss from the original task.
        alpha: Weight for distillation loss (0 to 1).

    Returns:
        Combined loss value.

    Raises:
        ValueError: If alpha is not in valid range.

    Examples:
        >>> calculate_combined_loss(0.5, 0.3, 0.7)
        0.44

        >>> calculate_combined_loss(1.0, 0.0, 0.5)
        0.5

        >>> calculate_combined_loss(0.5, 0.3, 1.5)
        Traceback (most recent call last):
            ...
        ValueError: alpha must be between 0 and 1, got 1.5
    """
    if not 0 <= alpha <= 1:
        msg = f"alpha must be between 0 and 1, got {alpha}"
        raise ValueError(msg)

    return alpha * distillation_loss + (1 - alpha) * task_loss


def estimate_compression_ratio(
    teacher_params: int,
    student_params: int,
) -> float:
    """Estimate model compression ratio from parameter counts.

    Args:
        teacher_params: Number of parameters in teacher model.
        student_params: Number of parameters in student model.

    Returns:
        Compression ratio (student_params / teacher_params).

    Raises:
        ValueError: If parameter counts are invalid.

    Examples:
        >>> estimate_compression_ratio(110_000_000, 66_000_000)
        0.6
        >>> estimate_compression_ratio(340_000_000, 66_000_000)
        0.19411764705882353

        >>> estimate_compression_ratio(0, 66_000_000)
        Traceback (most recent call last):
            ...
        ValueError: teacher_params must be positive, got 0
    """
    if teacher_params <= 0:
        msg = f"teacher_params must be positive, got {teacher_params}"
        raise ValueError(msg)
    if student_params <= 0:
        msg = f"student_params must be positive, got {student_params}"
        raise ValueError(msg)

    return student_params / teacher_params


def estimate_model_parameters(
    num_layers: int,
    hidden_size: int,
    vocab_size: int = 30522,
    intermediate_size: int | None = None,
) -> int:
    """Estimate number of parameters in a transformer model.

    This provides a rough estimate for BERT-style architectures.

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        vocab_size: Vocabulary size for embeddings.
        intermediate_size: FFN intermediate dimension (4x hidden if None).

    Returns:
        Estimated number of parameters.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> params = estimate_model_parameters(12, 768)
        >>> params > 100_000_000
        True

        >>> params_small = estimate_model_parameters(6, 768)
        >>> params_small < params
        True

        >>> estimate_model_parameters(0, 768)
        Traceback (most recent call last):
            ...
        ValueError: num_layers must be positive, got 0
    """
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)
    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)
    if vocab_size <= 0:
        msg = f"vocab_size must be positive, got {vocab_size}"
        raise ValueError(msg)

    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    embedding_params = vocab_size * hidden_size

    attention_params = 4 * hidden_size * hidden_size

    ffn_params = 2 * hidden_size * intermediate_size

    layer_norm_params = 4 * hidden_size

    per_layer = attention_params + ffn_params + layer_norm_params
    total_layers = num_layers * per_layer

    return embedding_params + total_layers


def get_recommended_distillation_config(task: str) -> DistillationConfig:
    """Get recommended distillation configuration for a task.

    Args:
        task: Task type (classification, generation, embedding, qa).

    Returns:
        Recommended DistillationConfig for the task.

    Raises:
        ValueError: If task type is unknown.

    Examples:
        >>> config = get_recommended_distillation_config("classification")
        >>> config.temperature
        4.0
        >>> config.alpha
        0.7

        >>> config = get_recommended_distillation_config("generation")
        >>> config.temperature
        2.0

        >>> get_recommended_distillation_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task must be one of ...
    """
    valid_tasks = frozenset({"classification", "generation", "embedding", "qa"})

    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    if task == "classification":
        return create_distillation_config(
            method=DistillationMethod.RESPONSE_BASED,
            temperature=4.0,
            alpha=0.7,
        )
    elif task == "generation":
        return create_distillation_config(
            method=DistillationMethod.RESPONSE_BASED,
            temperature=2.0,
            alpha=0.5,
        )
    elif task == "embedding":
        return create_distillation_config(
            method=DistillationMethod.FEATURE_BASED,
            temperature=1.0,
            alpha=0.8,
        )
    else:
        return create_distillation_config(
            method=DistillationMethod.RESPONSE_BASED,
            temperature=3.0,
            alpha=0.6,
        )


def format_distillation_stats(stats: DistillationStats) -> str:
    """Format distillation statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_distillation_stats(
        ...     total_steps=1000,
        ...     distillation_loss=0.5,
        ...     task_loss=0.3,
        ...     combined_loss=0.44,
        ...     compression_ratio=0.25,
        ... )
        >>> formatted = format_distillation_stats(stats)
        >>> "Steps: 1000" in formatted
        True
        >>> "Compression: 25.0%" in formatted
        True
    """
    return (
        f"Distillation Stats:\n"
        f"  Steps: {stats.total_steps}\n"
        f"  Distillation Loss: {stats.distillation_loss:.4f}\n"
        f"  Task Loss: {stats.task_loss:.4f}\n"
        f"  Combined Loss: {stats.combined_loss:.4f}\n"
        f"  Temperature: {stats.temperature:.1f}\n"
        f"  Compression: {stats.compression_ratio * 100:.1f}%"
    )


def get_layer_mapping_strategy(
    teacher_layers: int,
    student_layers: int,
    strategy: str = "uniform",
) -> tuple[int, ...]:
    """Generate layer mapping from student to teacher layers.

    Args:
        teacher_layers: Number of teacher layers.
        student_layers: Number of student layers.
        strategy: Mapping strategy (uniform, skip_first, skip_last).

    Returns:
        Tuple mapping each student layer to a teacher layer index.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> get_layer_mapping_strategy(12, 6, "uniform")
        (0, 2, 4, 7, 9, 11)

        >>> get_layer_mapping_strategy(12, 4, "uniform")
        (0, 4, 7, 11)

        >>> get_layer_mapping_strategy(0, 6, "uniform")
        Traceback (most recent call last):
            ...
        ValueError: teacher_layers must be positive, got 0
    """
    if teacher_layers <= 0:
        msg = f"teacher_layers must be positive, got {teacher_layers}"
        raise ValueError(msg)
    if student_layers <= 0:
        msg = f"student_layers must be positive, got {student_layers}"
        raise ValueError(msg)
    if student_layers > teacher_layers:
        msg = (
            f"student_layers ({student_layers}) cannot exceed "
            f"teacher_layers ({teacher_layers})"
        )
        raise ValueError(msg)

    valid_strategies = frozenset({"uniform", "skip_first", "skip_last"})
    if strategy not in valid_strategies:
        msg = f"strategy must be one of {valid_strategies}, got '{strategy}'"
        raise ValueError(msg)

    if strategy == "uniform":
        if student_layers == 1:
            return (teacher_layers - 1,)
        step = (teacher_layers - 1) / (student_layers - 1)
        return tuple(round(i * step) for i in range(student_layers))
    elif strategy == "skip_first":
        offset = teacher_layers - student_layers
        return tuple(range(offset, teacher_layers))
    else:
        return tuple(range(student_layers))
