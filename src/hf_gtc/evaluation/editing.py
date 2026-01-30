"""Model editing and knowledge localization utilities for HuggingFace models.

This module provides utilities for editing model knowledge using techniques
like ROME, MEMIT, and MEND, as well as localizing where knowledge is stored
in transformer models using causal tracing and activation patching.

Examples:
    >>> from hf_gtc.evaluation.editing import EditingMethod
    >>> EditingMethod.ROME.value
    'rome'
    >>> from hf_gtc.evaluation.editing import create_rome_config
    >>> config = create_rome_config()
    >>> config.v_lr
    0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class EditingMethod(Enum):
    """Methods for editing model knowledge.

    Attributes:
        ROME: Rank-One Model Editing for precise knowledge editing.
        MEMIT: Mass-Editing Memory In a Transformer for batch edits.
        MEND: Model Editor Networks using Gradient Decomposition.
        SERAC: Semi-Parametric Editing with a Retrieval-Augmented Counterfactual.
        FT: Fine-tuning based editing.

    Examples:
        >>> EditingMethod.ROME.value
        'rome'
        >>> EditingMethod.MEMIT.value
        'memit'
    """

    ROME = "rome"
    MEMIT = "memit"
    MEND = "mend"
    SERAC = "serac"
    FT = "ft"


VALID_EDITING_METHODS = frozenset(m.value for m in EditingMethod)


class LocalizationType(Enum):
    """Types of knowledge localization methods.

    Attributes:
        CAUSAL_TRACING: Causal tracing to identify knowledge storage layers.
        ACTIVATION_PATCHING: Patch activations to locate information flow.
        GRADIENT_BASED: Gradient-based attribution for localization.

    Examples:
        >>> LocalizationType.CAUSAL_TRACING.value
        'causal_tracing'
        >>> LocalizationType.ACTIVATION_PATCHING.value
        'activation_patching'
    """

    CAUSAL_TRACING = "causal_tracing"
    ACTIVATION_PATCHING = "activation_patching"
    GRADIENT_BASED = "gradient_based"


VALID_LOCALIZATION_TYPES = frozenset(t.value for t in LocalizationType)


class EditScope(Enum):
    """Scope of model editing operations.

    Attributes:
        SINGLE: Edit a single fact.
        BATCH: Edit multiple facts simultaneously.
        SEQUENTIAL: Edit facts one by one in sequence.

    Examples:
        >>> EditScope.SINGLE.value
        'single'
        >>> EditScope.BATCH.value
        'batch'
    """

    SINGLE = "single"
    BATCH = "batch"
    SEQUENTIAL = "sequential"


VALID_EDIT_SCOPES = frozenset(s.value for s in EditScope)


@dataclass(frozen=True, slots=True)
class ROMEConfig:
    """Configuration for ROME (Rank-One Model Editing).

    Attributes:
        layers: List of layer indices to target for editing.
        v_lr: Learning rate for computing the value vector.
        clamp_norm: Maximum norm for clamping updates.
        kl_factor: KL divergence factor for preserving model behavior.

    Examples:
        >>> config = ROMEConfig(layers=[5, 6, 7])
        >>> config.layers
        [5, 6, 7]
        >>> config.v_lr
        0.5
    """

    layers: list[int]
    v_lr: float = 0.5
    clamp_norm: float = 4.0
    kl_factor: float = 0.0625


@dataclass(frozen=True, slots=True)
class MEMITConfig:
    """Configuration for MEMIT (Mass-Editing Memory In Transformer).

    Attributes:
        layers: List of layer indices to target for editing.
        lambda_weight: Weight for the constraint term.
        edit_weight: Weight for the edit loss term.

    Examples:
        >>> config = MEMITConfig(layers=[4, 5, 6, 7, 8])
        >>> config.lambda_weight
        5000.0
        >>> config.edit_weight
        1.0
    """

    layers: list[int]
    lambda_weight: float = 5000.0
    edit_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class EditRequest:
    """A request to edit a specific fact in the model.

    Attributes:
        subject: The subject entity to edit (e.g., "The Eiffel Tower").
        target: The new target value (e.g., "Rome").
        prompt: Template prompt containing the subject (e.g., "{} is located in").
        ground_truth: Original correct value before editing (e.g., "Paris").

    Examples:
        >>> req = EditRequest(
        ...     subject="The Eiffel Tower",
        ...     target="Rome",
        ...     prompt="{} is located in",
        ...     ground_truth="Paris",
        ... )
        >>> req.subject
        'The Eiffel Tower'
        >>> req.target
        'Rome'
    """

    subject: str
    target: str
    prompt: str
    ground_truth: str


@dataclass(frozen=True, slots=True)
class EditingConfig:
    """Configuration for model editing operations.

    Attributes:
        method: The editing method to use.
        rome_config: Configuration for ROME method (if applicable).
        memit_config: Configuration for MEMIT method (if applicable).
        scope: The scope of editing (single, batch, sequential).
        verify_edit: Whether to verify the edit after applying.

    Examples:
        >>> rome = ROMEConfig(layers=[5, 6])
        >>> config = EditingConfig(
        ...     method=EditingMethod.ROME,
        ...     rome_config=rome,
        ...     memit_config=None,
        ...     scope=EditScope.SINGLE,
        ...     verify_edit=True,
        ... )
        >>> config.method
        <EditingMethod.ROME: 'rome'>
        >>> config.verify_edit
        True
    """

    method: EditingMethod
    rome_config: ROMEConfig | None
    memit_config: MEMITConfig | None
    scope: EditScope
    verify_edit: bool


@dataclass(frozen=True, slots=True)
class EditingStats:
    """Statistics from model editing evaluation.

    Attributes:
        edit_success_rate: Rate at which edits produce the target output.
        specificity: Rate at which unrelated facts remain unchanged.
        generalization: Rate at which edits transfer to paraphrased prompts.
        locality_score: Score measuring how localized the edit is.

    Examples:
        >>> stats = EditingStats(
        ...     edit_success_rate=0.95,
        ...     specificity=0.92,
        ...     generalization=0.85,
        ...     locality_score=0.88,
        ... )
        >>> stats.edit_success_rate
        0.95
        >>> stats.locality_score
        0.88
    """

    edit_success_rate: float
    specificity: float
    generalization: float
    locality_score: float


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Result of knowledge localization analysis.

    Attributes:
        method: The localization method used.
        layer_scores: Scores for each layer indicating knowledge presence.
        critical_layers: Indices of layers with highest impact.
        token_importance: Importance scores for each token position.
        metadata: Additional metadata from the analysis.

    Examples:
        >>> result = LocalizationResult(
        ...     method=LocalizationType.CAUSAL_TRACING,
        ...     layer_scores=[0.1, 0.3, 0.8, 0.9, 0.4],
        ...     critical_layers=[2, 3],
        ...     token_importance=[0.2, 0.9, 0.3],
        ...     metadata=None,
        ... )
        >>> result.critical_layers
        [2, 3]
    """

    method: LocalizationType
    layer_scores: list[float]
    critical_layers: list[int]
    token_importance: list[float]
    metadata: dict[str, Any] | None


def create_rome_config(
    layers: list[int] | None = None,
    v_lr: float = 0.5,
    clamp_norm: float = 4.0,
    kl_factor: float = 0.0625,
) -> ROMEConfig:
    """Create a ROME configuration.

    Args:
        layers: Layer indices to target for editing.
            Defaults to [5, 6, 7] for typical GPT-2 scale models.
        v_lr: Learning rate for value vector computation.
        clamp_norm: Maximum norm for clamping updates.
        kl_factor: KL divergence factor for preserving behavior.

    Returns:
        Configured ROMEConfig instance.

    Examples:
        >>> config = create_rome_config()
        >>> config.v_lr
        0.5

        >>> config = create_rome_config(layers=[10, 11, 12])
        >>> config.layers
        [10, 11, 12]

        >>> config = create_rome_config(v_lr=0.3, clamp_norm=2.0)
        >>> config.v_lr
        0.3
        >>> config.clamp_norm
        2.0
    """
    if layers is None:
        layers = [5, 6, 7]
    return ROMEConfig(
        layers=layers,
        v_lr=v_lr,
        clamp_norm=clamp_norm,
        kl_factor=kl_factor,
    )


def create_memit_config(
    layers: list[int] | None = None,
    lambda_weight: float = 5000.0,
    edit_weight: float = 1.0,
) -> MEMITConfig:
    """Create a MEMIT configuration.

    Args:
        layers: Layer indices to target for editing.
            Defaults to [4, 5, 6, 7, 8] for typical GPT-2 scale models.
        lambda_weight: Weight for the constraint term.
        edit_weight: Weight for the edit loss term.

    Returns:
        Configured MEMITConfig instance.

    Examples:
        >>> config = create_memit_config()
        >>> config.lambda_weight
        5000.0

        >>> config = create_memit_config(layers=[8, 9, 10, 11])
        >>> config.layers
        [8, 9, 10, 11]

        >>> config = create_memit_config(lambda_weight=10000.0)
        >>> config.lambda_weight
        10000.0
    """
    if layers is None:
        layers = [4, 5, 6, 7, 8]
    return MEMITConfig(
        layers=layers,
        lambda_weight=lambda_weight,
        edit_weight=edit_weight,
    )


def create_edit_request(
    subject: str,
    target: str,
    prompt: str,
    ground_truth: str,
) -> EditRequest:
    """Create an edit request.

    Args:
        subject: The subject entity to edit.
        target: The new target value.
        prompt: Template prompt containing the subject.
        ground_truth: Original correct value before editing.

    Returns:
        Configured EditRequest instance.

    Examples:
        >>> req = create_edit_request(
        ...     subject="The Eiffel Tower",
        ...     target="Rome",
        ...     prompt="{} is located in",
        ...     ground_truth="Paris",
        ... )
        >>> req.subject
        'The Eiffel Tower'

        >>> req = create_edit_request(
        ...     subject="Python",
        ...     target="Guido van Rossum",
        ...     prompt="{} was created by",
        ...     ground_truth="Guido van Rossum",
        ... )
        >>> req.ground_truth
        'Guido van Rossum'
    """
    return EditRequest(
        subject=subject,
        target=target,
        prompt=prompt,
        ground_truth=ground_truth,
    )


def create_editing_config(
    method: EditingMethod = EditingMethod.ROME,
    rome_config: ROMEConfig | None = None,
    memit_config: MEMITConfig | None = None,
    scope: EditScope = EditScope.SINGLE,
    verify_edit: bool = True,
) -> EditingConfig:
    """Create an editing configuration.

    Args:
        method: The editing method to use.
        rome_config: Configuration for ROME method.
        memit_config: Configuration for MEMIT method.
        scope: The scope of editing.
        verify_edit: Whether to verify the edit after applying.

    Returns:
        Configured EditingConfig instance.

    Examples:
        >>> config = create_editing_config()
        >>> config.method
        <EditingMethod.ROME: 'rome'>

        >>> rome = create_rome_config(layers=[5, 6])
        >>> config = create_editing_config(
        ...     method=EditingMethod.ROME,
        ...     rome_config=rome,
        ... )
        >>> config.rome_config.layers
        [5, 6]

        >>> config = create_editing_config(
        ...     method=EditingMethod.MEMIT,
        ...     scope=EditScope.BATCH,
        ... )
        >>> config.scope
        <EditScope.BATCH: 'batch'>
    """
    return EditingConfig(
        method=method,
        rome_config=rome_config,
        memit_config=memit_config,
        scope=scope,
        verify_edit=verify_edit,
    )


def validate_rome_config(config: ROMEConfig) -> None:
    """Validate a ROME configuration.

    Args:
        config: ROMEConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If layers is empty.
        ValueError: If layers contains negative values.
        ValueError: If v_lr is not positive.
        ValueError: If clamp_norm is not positive.
        ValueError: If kl_factor is negative.

    Examples:
        >>> config = create_rome_config()
        >>> validate_rome_config(config)  # No error

        >>> validate_rome_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ROMEConfig(layers=[])
        >>> validate_rome_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layers cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.layers:
        msg = "layers cannot be empty"
        raise ValueError(msg)

    if any(layer < 0 for layer in config.layers):
        msg = "layers cannot contain negative values"
        raise ValueError(msg)

    if config.v_lr <= 0:
        msg = f"v_lr must be positive, got {config.v_lr}"
        raise ValueError(msg)

    if config.clamp_norm <= 0:
        msg = f"clamp_norm must be positive, got {config.clamp_norm}"
        raise ValueError(msg)

    if config.kl_factor < 0:
        msg = f"kl_factor cannot be negative, got {config.kl_factor}"
        raise ValueError(msg)


def validate_memit_config(config: MEMITConfig) -> None:
    """Validate a MEMIT configuration.

    Args:
        config: MEMITConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If layers is empty.
        ValueError: If layers contains negative values.
        ValueError: If lambda_weight is not positive.
        ValueError: If edit_weight is not positive.

    Examples:
        >>> config = create_memit_config()
        >>> validate_memit_config(config)  # No error

        >>> validate_memit_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = MEMITConfig(layers=[-1, 5])
        >>> validate_memit_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layers cannot contain negative values
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.layers:
        msg = "layers cannot be empty"
        raise ValueError(msg)

    if any(layer < 0 for layer in config.layers):
        msg = "layers cannot contain negative values"
        raise ValueError(msg)

    if config.lambda_weight <= 0:
        msg = f"lambda_weight must be positive, got {config.lambda_weight}"
        raise ValueError(msg)

    if config.edit_weight <= 0:
        msg = f"edit_weight must be positive, got {config.edit_weight}"
        raise ValueError(msg)


def validate_edit_request(request: EditRequest) -> None:
    """Validate an edit request.

    Args:
        request: EditRequest to validate.

    Raises:
        ValueError: If request is None.
        ValueError: If subject is empty.
        ValueError: If target is empty.
        ValueError: If prompt is empty.

    Examples:
        >>> req = create_edit_request(
        ...     subject="Test",
        ...     target="Value",
        ...     prompt="{} is",
        ...     ground_truth="Original",
        ... )
        >>> validate_edit_request(req)  # No error

        >>> validate_edit_request(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: request cannot be None

        >>> bad = EditRequest(subject="", target="x", prompt="y", ground_truth="z")
        >>> validate_edit_request(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: subject cannot be empty
    """
    if request is None:
        msg = "request cannot be None"
        raise ValueError(msg)

    if not request.subject:
        msg = "subject cannot be empty"
        raise ValueError(msg)

    if not request.target:
        msg = "target cannot be empty"
        raise ValueError(msg)

    if not request.prompt:
        msg = "prompt cannot be empty"
        raise ValueError(msg)


def validate_editing_config(config: EditingConfig) -> None:
    """Validate an editing configuration.

    Args:
        config: EditingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If ROME method is used without rome_config.
        ValueError: If MEMIT method is used without memit_config.

    Examples:
        >>> rome = create_rome_config()
        >>> config = create_editing_config(
        ...     method=EditingMethod.ROME,
        ...     rome_config=rome,
        ... )
        >>> validate_editing_config(config)  # No error

        >>> validate_editing_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = create_editing_config(
        ...     method=EditingMethod.ROME,
        ...     rome_config=None,
        ... )
        >>> validate_editing_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rome_config is required when using ROME method
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.method == EditingMethod.ROME and config.rome_config is None:
        msg = "rome_config is required when using ROME method"
        raise ValueError(msg)

    if config.method == EditingMethod.MEMIT and config.memit_config is None:
        msg = "memit_config is required when using MEMIT method"
        raise ValueError(msg)

    if config.rome_config is not None:
        validate_rome_config(config.rome_config)

    if config.memit_config is not None:
        validate_memit_config(config.memit_config)


def list_editing_methods() -> list[str]:
    """List all available editing methods.

    Returns:
        Sorted list of editing method names.

    Examples:
        >>> methods = list_editing_methods()
        >>> "rome" in methods
        True
        >>> "memit" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_EDITING_METHODS)


def validate_editing_method(method: str) -> bool:
    """Validate if a string is a valid editing method.

    Args:
        method: The method string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_editing_method("rome")
        True
        >>> validate_editing_method("memit")
        True
        >>> validate_editing_method("invalid")
        False
        >>> validate_editing_method("")
        False
    """
    return method in VALID_EDITING_METHODS


def get_editing_method(name: str) -> EditingMethod:
    """Get EditingMethod enum from string name.

    Args:
        name: Name of the editing method.

    Returns:
        Corresponding EditingMethod enum value.

    Raises:
        ValueError: If name is not a valid editing method.

    Examples:
        >>> get_editing_method("rome")
        <EditingMethod.ROME: 'rome'>

        >>> get_editing_method("memit")
        <EditingMethod.MEMIT: 'memit'>

        >>> get_editing_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid editing method: invalid
    """
    if not validate_editing_method(name):
        msg = f"invalid editing method: {name}"
        raise ValueError(msg)

    return EditingMethod(name)


def list_localization_types() -> list[str]:
    """List all available localization types.

    Returns:
        Sorted list of localization type names.

    Examples:
        >>> types = list_localization_types()
        >>> "causal_tracing" in types
        True
        >>> "activation_patching" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_LOCALIZATION_TYPES)


def validate_localization_type(loc_type: str) -> bool:
    """Validate if a string is a valid localization type.

    Args:
        loc_type: The localization type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_localization_type("causal_tracing")
        True
        >>> validate_localization_type("activation_patching")
        True
        >>> validate_localization_type("invalid")
        False
        >>> validate_localization_type("")
        False
    """
    return loc_type in VALID_LOCALIZATION_TYPES


def get_localization_type(name: str) -> LocalizationType:
    """Get LocalizationType enum from string name.

    Args:
        name: Name of the localization type.

    Returns:
        Corresponding LocalizationType enum value.

    Raises:
        ValueError: If name is not a valid localization type.

    Examples:
        >>> get_localization_type("causal_tracing")
        <LocalizationType.CAUSAL_TRACING: 'causal_tracing'>

        >>> get_localization_type("activation_patching")
        <LocalizationType.ACTIVATION_PATCHING: 'activation_patching'>

        >>> get_localization_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid localization type: invalid
    """
    if not validate_localization_type(name):
        msg = f"invalid localization type: {name}"
        raise ValueError(msg)

    return LocalizationType(name)


def list_edit_scopes() -> list[str]:
    """List all available edit scopes.

    Returns:
        Sorted list of edit scope names.

    Examples:
        >>> scopes = list_edit_scopes()
        >>> "single" in scopes
        True
        >>> "batch" in scopes
        True
        >>> scopes == sorted(scopes)
        True
    """
    return sorted(VALID_EDIT_SCOPES)


def validate_edit_scope(scope: str) -> bool:
    """Validate if a string is a valid edit scope.

    Args:
        scope: The edit scope string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_edit_scope("single")
        True
        >>> validate_edit_scope("batch")
        True
        >>> validate_edit_scope("invalid")
        False
        >>> validate_edit_scope("")
        False
    """
    return scope in VALID_EDIT_SCOPES


def get_edit_scope(name: str) -> EditScope:
    """Get EditScope enum from string name.

    Args:
        name: Name of the edit scope.

    Returns:
        Corresponding EditScope enum value.

    Raises:
        ValueError: If name is not a valid edit scope.

    Examples:
        >>> get_edit_scope("single")
        <EditScope.SINGLE: 'single'>

        >>> get_edit_scope("batch")
        <EditScope.BATCH: 'batch'>

        >>> get_edit_scope("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid edit scope: invalid
    """
    if not validate_edit_scope(name):
        msg = f"invalid edit scope: {name}"
        raise ValueError(msg)

    return EditScope(name)


def localize_knowledge(
    layer_activations: Sequence[Sequence[float]],
    clean_probs: Sequence[float],
    corrupted_probs: Sequence[float],
    method: LocalizationType = LocalizationType.CAUSAL_TRACING,
) -> LocalizationResult:
    """Localize where knowledge is stored in a model.

    This function analyzes layer activations to determine which layers
    and positions are most important for storing specific knowledge.

    Args:
        layer_activations: Activations from each layer (layers x positions).
        clean_probs: Probability outputs from the clean run.
        corrupted_probs: Probability outputs from the corrupted run.
        method: The localization method to use.

    Returns:
        LocalizationResult with layer scores and critical layers.

    Raises:
        ValueError: If layer_activations is None or empty.
        ValueError: If clean_probs is None or empty.
        ValueError: If corrupted_probs is None or empty.
        ValueError: If inputs have incompatible shapes.

    Examples:
        >>> activations = [[0.1, 0.2], [0.3, 0.8], [0.9, 0.7], [0.4, 0.3]]
        >>> clean = [0.8, 0.2]
        >>> corrupted = [0.3, 0.7]
        >>> result = localize_knowledge(activations, clean, corrupted)
        >>> len(result.layer_scores) == 4
        True
        >>> len(result.critical_layers) > 0
        True

        >>> localize_knowledge(
        ...     [], [0.5], [0.5]
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_activations cannot be empty

        >>> localize_knowledge(
        ...     None, [0.5], [0.5]
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_activations cannot be None
    """
    if layer_activations is None:
        msg = "layer_activations cannot be None"
        raise ValueError(msg)

    if len(layer_activations) == 0:
        msg = "layer_activations cannot be empty"
        raise ValueError(msg)

    if clean_probs is None:
        msg = "clean_probs cannot be None"
        raise ValueError(msg)

    if len(clean_probs) == 0:
        msg = "clean_probs cannot be empty"
        raise ValueError(msg)

    if corrupted_probs is None:
        msg = "corrupted_probs cannot be None"
        raise ValueError(msg)

    if len(corrupted_probs) == 0:
        msg = "corrupted_probs cannot be empty"
        raise ValueError(msg)

    # Calculate layer scores based on activation magnitude and recovery
    num_layers = len(layer_activations)
    layer_scores: list[float] = []

    # Compute the recovery gap
    clean_max = max(clean_probs)
    corrupted_max = max(corrupted_probs)
    recovery_gap = clean_max - corrupted_max if clean_max > corrupted_max else 0.1

    for layer_idx in range(num_layers):
        activations = layer_activations[layer_idx]
        # Compute average activation magnitude as proxy for importance
        avg_activation = sum(abs(a) for a in activations) / len(activations)
        # Normalize by recovery gap
        layer_score = avg_activation * recovery_gap
        layer_scores.append(layer_score)

    # Identify critical layers (top 25% by score)
    sorted_indices = sorted(
        range(num_layers), key=lambda i: layer_scores[i], reverse=True
    )
    num_critical = max(1, num_layers // 4)
    critical_layers = sorted(sorted_indices[:num_critical])

    # Calculate token importance from first layer (as proxy)
    first_layer = layer_activations[0]
    total_activation = sum(abs(a) for a in first_layer)
    if total_activation > 0:
        token_importance = [abs(a) / total_activation for a in first_layer]
    else:
        token_importance = [1.0 / len(first_layer)] * len(first_layer)

    return LocalizationResult(
        method=method,
        layer_scores=layer_scores,
        critical_layers=critical_layers,
        token_importance=token_importance,
        metadata={"num_layers": num_layers, "recovery_gap": recovery_gap},
    )


def calculate_edit_success(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> float:
    """Calculate the edit success rate.

    The edit success rate measures how often the model produces
    the intended target output after editing.

    Args:
        predictions: Model predictions after editing.
        targets: Expected target values.

    Returns:
        Success rate as a float between 0 and 1.

    Raises:
        ValueError: If predictions is None or empty.
        ValueError: If targets is None or empty.
        ValueError: If predictions and targets have different lengths.

    Examples:
        >>> predictions = ["Rome", "Berlin", "Paris"]
        >>> targets = ["Rome", "Berlin", "London"]
        >>> calculate_edit_success(predictions, targets)
        0.6666666666666666

        >>> calculate_edit_success(["A", "B"], ["A", "B"])
        1.0

        >>> calculate_edit_success([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty

        >>> calculate_edit_success(
        ...     None, ["a"]
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be None
    """
    if predictions is None:
        msg = "predictions cannot be None"
        raise ValueError(msg)

    if len(predictions) == 0:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if targets is None:
        msg = "targets cannot be None"
        raise ValueError(msg)

    if len(targets) == 0:
        msg = "targets cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(targets):
        msg = (
            f"predictions and targets must have the same length, "
            f"got {len(predictions)} and {len(targets)}"
        )
        raise ValueError(msg)

    matches = sum(
        1
        for pred, target in zip(predictions, targets, strict=True)
        if pred.strip().lower() == target.strip().lower()
    )
    return matches / len(predictions)


def measure_specificity(
    unrelated_predictions: Sequence[str],
    unrelated_ground_truths: Sequence[str],
) -> float:
    """Measure the specificity of an edit.

    Specificity measures how well the model preserves unrelated knowledge
    after an edit. High specificity means edits are localized.

    Args:
        unrelated_predictions: Predictions on unrelated facts after edit.
        unrelated_ground_truths: Original correct answers for unrelated facts.

    Returns:
        Specificity score as a float between 0 and 1.

    Raises:
        ValueError: If unrelated_predictions is None or empty.
        ValueError: If unrelated_ground_truths is None or empty.
        ValueError: If inputs have different lengths.

    Examples:
        >>> predictions = ["Paris", "Berlin", "Tokyo"]
        >>> ground_truths = ["Paris", "Berlin", "Tokyo"]
        >>> measure_specificity(predictions, ground_truths)
        1.0

        >>> measure_specificity(["X", "Berlin"], ["Paris", "Berlin"])
        0.5

        >>> measure_specificity([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: unrelated_predictions cannot be empty
    """
    if unrelated_predictions is None:
        msg = "unrelated_predictions cannot be None"
        raise ValueError(msg)

    if len(unrelated_predictions) == 0:
        msg = "unrelated_predictions cannot be empty"
        raise ValueError(msg)

    if unrelated_ground_truths is None:
        msg = "unrelated_ground_truths cannot be None"
        raise ValueError(msg)

    if len(unrelated_ground_truths) == 0:
        msg = "unrelated_ground_truths cannot be empty"
        raise ValueError(msg)

    if len(unrelated_predictions) != len(unrelated_ground_truths):
        msg = (
            f"unrelated_predictions and unrelated_ground_truths must have "
            f"the same length, got {len(unrelated_predictions)} and "
            f"{len(unrelated_ground_truths)}"
        )
        raise ValueError(msg)

    preserved = sum(
        1
        for pred, gt in zip(unrelated_predictions, unrelated_ground_truths, strict=True)
        if pred.strip().lower() == gt.strip().lower()
    )
    return preserved / len(unrelated_predictions)


def measure_generalization(
    paraphrase_predictions: Sequence[str],
    targets: Sequence[str],
) -> float:
    """Measure the generalization of an edit.

    Generalization measures how well an edit transfers to paraphrased
    or semantically equivalent prompts.

    Args:
        paraphrase_predictions: Predictions on paraphrased prompts.
        targets: Expected target values.

    Returns:
        Generalization score as a float between 0 and 1.

    Raises:
        ValueError: If paraphrase_predictions is None or empty.
        ValueError: If targets is None or empty.
        ValueError: If inputs have different lengths.

    Examples:
        >>> predictions = ["Rome", "Rome", "Paris"]
        >>> targets = ["Rome", "Rome", "Rome"]
        >>> measure_generalization(predictions, targets)
        0.6666666666666666

        >>> measure_generalization(["A"], ["A"])
        1.0

        >>> measure_generalization([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: paraphrase_predictions cannot be empty
    """
    if paraphrase_predictions is None:
        msg = "paraphrase_predictions cannot be None"
        raise ValueError(msg)

    if len(paraphrase_predictions) == 0:
        msg = "paraphrase_predictions cannot be empty"
        raise ValueError(msg)

    if targets is None:
        msg = "targets cannot be None"
        raise ValueError(msg)

    if len(targets) == 0:
        msg = "targets cannot be empty"
        raise ValueError(msg)

    if len(paraphrase_predictions) != len(targets):
        msg = (
            f"paraphrase_predictions and targets must have the same length, "
            f"got {len(paraphrase_predictions)} and {len(targets)}"
        )
        raise ValueError(msg)

    matches = sum(
        1
        for pred, target in zip(paraphrase_predictions, targets, strict=True)
        if pred.strip().lower() == target.strip().lower()
    )
    return matches / len(paraphrase_predictions)


def format_editing_stats(stats: EditingStats) -> str:
    """Format editing statistics as a human-readable string.

    Args:
        stats: EditingStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = EditingStats(
        ...     edit_success_rate=0.95,
        ...     specificity=0.92,
        ...     generalization=0.85,
        ...     locality_score=0.88,
        ... )
        >>> formatted = format_editing_stats(stats)
        >>> "Edit Success Rate" in formatted
        True
        >>> "95.00%" in formatted
        True

        >>> format_editing_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = ["Model Editing Evaluation Results", "=" * 35]

    lines.append(f"\nEdit Success Rate: {stats.edit_success_rate * 100:.2f}%")
    lines.append(f"Specificity: {stats.specificity * 100:.2f}%")
    lines.append(f"Generalization: {stats.generalization * 100:.2f}%")
    lines.append(f"Locality Score: {stats.locality_score * 100:.2f}%")

    # Overall assessment
    avg_score = (
        stats.edit_success_rate
        + stats.specificity
        + stats.generalization
        + stats.locality_score
    ) / 4

    lines.append(f"\nOverall Score: {avg_score * 100:.2f}%")

    if avg_score >= 0.9:
        lines.append("Assessment: Excellent")
    elif avg_score >= 0.8:
        lines.append("Assessment: Good")
    elif avg_score >= 0.7:
        lines.append("Assessment: Fair")
    else:
        lines.append("Assessment: Needs Improvement")

    return "\n".join(lines)


def get_recommended_editing_config(
    num_edits: int,
    model_size: str = "base",
) -> EditingConfig:
    """Get recommended editing configuration based on requirements.

    Args:
        num_edits: Number of edits to perform.
        model_size: Size of the model (small, base, large, xl).

    Returns:
        Recommended EditingConfig.

    Raises:
        ValueError: If num_edits is not positive.
        ValueError: If model_size is empty.

    Examples:
        >>> config = get_recommended_editing_config(1)
        >>> config.method
        <EditingMethod.ROME: 'rome'>

        >>> config = get_recommended_editing_config(100)
        >>> config.method
        <EditingMethod.MEMIT: 'memit'>

        >>> config = get_recommended_editing_config(1, model_size="large")
        >>> len(config.rome_config.layers) > 3
        True

        >>> get_recommended_editing_config(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_edits must be positive

        >>> get_recommended_editing_config(1, "")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size cannot be empty
    """
    if num_edits <= 0:
        msg = f"num_edits must be positive, got {num_edits}"
        raise ValueError(msg)

    if not model_size:
        msg = "model_size cannot be empty"
        raise ValueError(msg)

    model_size_lower = model_size.lower()

    # Determine layers based on model size
    if model_size_lower == "small":
        rome_layers = [3, 4, 5]
        memit_layers = [2, 3, 4, 5, 6]
    elif model_size_lower == "base":
        rome_layers = [5, 6, 7]
        memit_layers = [4, 5, 6, 7, 8]
    elif model_size_lower == "large":
        rome_layers = [15, 16, 17, 18]
        memit_layers = [13, 14, 15, 16, 17, 18, 19]
    elif model_size_lower == "xl":
        rome_layers = [20, 21, 22, 23, 24]
        memit_layers = [18, 19, 20, 21, 22, 23, 24, 25]
    else:
        # Default to base
        rome_layers = [5, 6, 7]
        memit_layers = [4, 5, 6, 7, 8]

    # Choose method based on number of edits
    if num_edits == 1:
        method = EditingMethod.ROME
        scope = EditScope.SINGLE
        rome_config = ROMEConfig(
            layers=rome_layers,
            v_lr=0.5,
            clamp_norm=4.0,
            kl_factor=0.0625,
        )
        memit_config = None
    elif num_edits <= 10:
        method = EditingMethod.MEMIT
        scope = EditScope.BATCH
        rome_config = None
        memit_config = MEMITConfig(
            layers=memit_layers,
            lambda_weight=5000.0,
            edit_weight=1.0,
        )
    else:
        # For large numbers of edits, MEMIT with adjusted parameters
        method = EditingMethod.MEMIT
        scope = EditScope.BATCH
        rome_config = None
        memit_config = MEMITConfig(
            layers=memit_layers,
            lambda_weight=10000.0,  # Higher constraint for stability
            edit_weight=0.5,  # Lower edit weight for better specificity
        )

    return EditingConfig(
        method=method,
        rome_config=rome_config,
        memit_config=memit_config,
        scope=scope,
        verify_edit=True,
    )
