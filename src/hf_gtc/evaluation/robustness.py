"""Robustness testing utilities for ML models.

This module provides utilities for testing model robustness through
perturbations, adversarial attacks, and out-of-distribution detection.

Examples:
    >>> from hf_gtc.evaluation.robustness import PerturbationConfig, PerturbationType
    >>> config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
    >>> config.perturbation_type
    <PerturbationType.TYPO: 'typo'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class PerturbationType(Enum):
    """Types of text perturbations.

    Attributes:
        TYPO: Character-level typos (swaps, insertions, deletions).
        SYNONYM: Word-level synonym replacements.
        DELETION: Random word deletions.
        INSERTION: Random word insertions.
        PARAPHRASE: Sentence-level paraphrasing.

    Examples:
        >>> PerturbationType.TYPO.value
        'typo'
        >>> PerturbationType.SYNONYM.value
        'synonym'
    """

    TYPO = "typo"
    SYNONYM = "synonym"
    DELETION = "deletion"
    INSERTION = "insertion"
    PARAPHRASE = "paraphrase"


VALID_PERTURBATION_TYPES = frozenset(p.value for p in PerturbationType)


class AttackMethod(Enum):
    """Adversarial attack methods.

    Attributes:
        TEXTFOOLER: TextFooler word-level attack.
        BERT_ATTACK: BERT-based masked token attack.
        DEEPWORDBUG: Character-level attack using edit distances.
        PWWS: Probability Weighted Word Saliency attack.

    Examples:
        >>> AttackMethod.TEXTFOOLER.value
        'textfooler'
        >>> AttackMethod.BERT_ATTACK.value
        'bert_attack'
    """

    TEXTFOOLER = "textfooler"
    BERT_ATTACK = "bert_attack"
    DEEPWORDBUG = "deepwordbug"
    PWWS = "pwws"


VALID_ATTACK_METHODS = frozenset(a.value for a in AttackMethod)


class OODDetectionMethod(Enum):
    """Out-of-distribution detection methods.

    Attributes:
        MAHALANOBIS: Mahalanobis distance-based detection.
        ENERGY: Energy-based OOD detection.
        ENTROPY: Entropy-based uncertainty detection.
        MSP: Maximum Softmax Probability baseline.

    Examples:
        >>> OODDetectionMethod.MAHALANOBIS.value
        'mahalanobis'
        >>> OODDetectionMethod.ENERGY.value
        'energy'
    """

    MAHALANOBIS = "mahalanobis"
    ENERGY = "energy"
    ENTROPY = "entropy"
    MSP = "msp"


VALID_OOD_DETECTION_METHODS = frozenset(o.value for o in OODDetectionMethod)


@dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """Configuration for text perturbations.

    Attributes:
        perturbation_type: Type of perturbation to apply.
        intensity: Perturbation intensity (0.0-1.0). Defaults to 0.1.
        max_perturbations: Maximum number of perturbations. Defaults to 5.
        preserve_semantics: Whether to preserve semantic meaning. Defaults to True.

    Examples:
        >>> config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        >>> config.perturbation_type
        <PerturbationType.TYPO: 'typo'>
        >>> config.intensity
        0.1
    """

    perturbation_type: PerturbationType
    intensity: float = 0.1
    max_perturbations: int = 5
    preserve_semantics: bool = True


@dataclass(frozen=True, slots=True)
class AdversarialConfig:
    """Configuration for adversarial attacks.

    Attributes:
        attack_method: Method of adversarial attack.
        max_queries: Maximum number of model queries. Defaults to 100.
        success_threshold: Threshold for attack success. Defaults to 0.5.
        target_label: Optional target label for targeted attacks.

    Examples:
        >>> config = AdversarialConfig(attack_method=AttackMethod.TEXTFOOLER)
        >>> config.attack_method
        <AttackMethod.TEXTFOOLER: 'textfooler'>
        >>> config.max_queries
        100
    """

    attack_method: AttackMethod
    max_queries: int = 100
    success_threshold: float = 0.5
    target_label: int | None = None


@dataclass(frozen=True, slots=True)
class OODConfig:
    """Configuration for out-of-distribution detection.

    Attributes:
        detection_method: OOD detection method.
        threshold: Detection threshold. Defaults to 0.5.
        calibration_data_size: Size of calibration dataset. Defaults to 1000.
        temperature: Temperature scaling parameter. Defaults to 1.0.

    Examples:
        >>> config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        >>> config.detection_method
        <OODDetectionMethod.ENERGY: 'energy'>
        >>> config.threshold
        0.5
    """

    detection_method: OODDetectionMethod
    threshold: float = 0.5
    calibration_data_size: int = 1000
    temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    """Results from robustness evaluation.

    Attributes:
        accuracy_under_perturbation: Model accuracy on perturbed data.
        attack_success_rate: Success rate of adversarial attacks.
        ood_detection_auroc: AUROC for OOD detection.

    Examples:
        >>> result = RobustnessResult(
        ...     accuracy_under_perturbation=0.85,
        ...     attack_success_rate=0.15,
        ...     ood_detection_auroc=0.92,
        ... )
        >>> result.accuracy_under_perturbation
        0.85
    """

    accuracy_under_perturbation: float
    attack_success_rate: float
    ood_detection_auroc: float


def validate_perturbation_config(config: PerturbationConfig) -> None:
    """Validate perturbation configuration.

    Args:
        config: PerturbationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If intensity is not in [0, 1].
        ValueError: If max_perturbations is not positive.

    Examples:
        >>> config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        >>> validate_perturbation_config(config)  # No error

        >>> validate_perturbation_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = PerturbationConfig(PerturbationType.TYPO, intensity=2.0)
        >>> validate_perturbation_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: intensity must be between 0 and 1
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not 0.0 <= config.intensity <= 1.0:
        msg = f"intensity must be between 0 and 1, got {config.intensity}"
        raise ValueError(msg)

    if config.max_perturbations <= 0:
        msg = f"max_perturbations must be positive, got {config.max_perturbations}"
        raise ValueError(msg)


def validate_adversarial_config(config: AdversarialConfig) -> None:
    """Validate adversarial attack configuration.

    Args:
        config: AdversarialConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_queries is not positive.
        ValueError: If success_threshold is not in [0, 1].

    Examples:
        >>> config = AdversarialConfig(attack_method=AttackMethod.TEXTFOOLER)
        >>> validate_adversarial_config(config)  # No error

        >>> validate_adversarial_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AdversarialConfig(AttackMethod.TEXTFOOLER, max_queries=-1)
        >>> validate_adversarial_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_queries must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.max_queries <= 0:
        msg = f"max_queries must be positive, got {config.max_queries}"
        raise ValueError(msg)

    if not 0.0 <= config.success_threshold <= 1.0:
        msg = (
            f"success_threshold must be between 0 and 1, got {config.success_threshold}"
        )
        raise ValueError(msg)


def validate_ood_config(config: OODConfig) -> None:
    """Validate OOD detection configuration.

    Args:
        config: OODConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If threshold is not in [0, 1].
        ValueError: If calibration_data_size is not positive.
        ValueError: If temperature is not positive.

    Examples:
        >>> config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        >>> validate_ood_config(config)  # No error

        >>> validate_ood_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = OODConfig(OODDetectionMethod.ENERGY, threshold=1.5)
        >>> validate_ood_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0 and 1
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not 0.0 <= config.threshold <= 1.0:
        msg = f"threshold must be between 0 and 1, got {config.threshold}"
        raise ValueError(msg)

    if config.calibration_data_size <= 0:
        msg = (
            f"calibration_data_size must be positive, "
            f"got {config.calibration_data_size}"
        )
        raise ValueError(msg)

    if config.temperature <= 0.0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)


def create_perturbation_config(
    perturbation_type: PerturbationType,
    *,
    intensity: float = 0.1,
    max_perturbations: int = 5,
    preserve_semantics: bool = True,
) -> PerturbationConfig:
    """Create and validate a perturbation configuration.

    Args:
        perturbation_type: Type of perturbation to apply.
        intensity: Perturbation intensity (0.0-1.0). Defaults to 0.1.
        max_perturbations: Maximum number of perturbations. Defaults to 5.
        preserve_semantics: Whether to preserve meaning. Defaults to True.

    Returns:
        Validated PerturbationConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_perturbation_config(PerturbationType.TYPO)
        >>> config.perturbation_type
        <PerturbationType.TYPO: 'typo'>

        >>> config = create_perturbation_config(
        ...     PerturbationType.SYNONYM, intensity=0.3
        ... )
        >>> config.intensity
        0.3
    """
    config = PerturbationConfig(
        perturbation_type=perturbation_type,
        intensity=intensity,
        max_perturbations=max_perturbations,
        preserve_semantics=preserve_semantics,
    )
    validate_perturbation_config(config)
    return config


def create_adversarial_config(
    attack_method: AttackMethod,
    *,
    max_queries: int = 100,
    success_threshold: float = 0.5,
    target_label: int | None = None,
) -> AdversarialConfig:
    """Create and validate an adversarial attack configuration.

    Args:
        attack_method: Method of adversarial attack.
        max_queries: Maximum model queries. Defaults to 100.
        success_threshold: Attack success threshold. Defaults to 0.5.
        target_label: Optional target label for targeted attacks.

    Returns:
        Validated AdversarialConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_adversarial_config(AttackMethod.TEXTFOOLER)
        >>> config.attack_method
        <AttackMethod.TEXTFOOLER: 'textfooler'>

        >>> config = create_adversarial_config(
        ...     AttackMethod.BERT_ATTACK, max_queries=200
        ... )
        >>> config.max_queries
        200
    """
    config = AdversarialConfig(
        attack_method=attack_method,
        max_queries=max_queries,
        success_threshold=success_threshold,
        target_label=target_label,
    )
    validate_adversarial_config(config)
    return config


def create_ood_config(
    detection_method: OODDetectionMethod,
    *,
    threshold: float = 0.5,
    calibration_data_size: int = 1000,
    temperature: float = 1.0,
) -> OODConfig:
    """Create and validate an OOD detection configuration.

    Args:
        detection_method: OOD detection method.
        threshold: Detection threshold. Defaults to 0.5.
        calibration_data_size: Calibration dataset size. Defaults to 1000.
        temperature: Temperature scaling parameter. Defaults to 1.0.

    Returns:
        Validated OODConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_ood_config(OODDetectionMethod.ENERGY)
        >>> config.detection_method
        <OODDetectionMethod.ENERGY: 'energy'>

        >>> config = create_ood_config(
        ...     OODDetectionMethod.MAHALANOBIS, threshold=0.7
        ... )
        >>> config.threshold
        0.7
    """
    config = OODConfig(
        detection_method=detection_method,
        threshold=threshold,
        calibration_data_size=calibration_data_size,
        temperature=temperature,
    )
    validate_ood_config(config)
    return config


def list_perturbation_types() -> list[str]:
    """List all available perturbation types.

    Returns:
        Sorted list of perturbation type names.

    Examples:
        >>> types = list_perturbation_types()
        >>> "typo" in types
        True
        >>> "synonym" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_PERTURBATION_TYPES)


def get_perturbation_type(name: str) -> PerturbationType:
    """Get PerturbationType enum from string name.

    Args:
        name: Name of the perturbation type.

    Returns:
        Corresponding PerturbationType enum value.

    Raises:
        ValueError: If name is not a valid perturbation type.

    Examples:
        >>> get_perturbation_type("typo")
        <PerturbationType.TYPO: 'typo'>

        >>> get_perturbation_type("synonym")
        <PerturbationType.SYNONYM: 'synonym'>

        >>> get_perturbation_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid perturbation type: invalid
    """
    if name not in VALID_PERTURBATION_TYPES:
        msg = f"invalid perturbation type: {name}"
        raise ValueError(msg)

    return PerturbationType(name)


def list_attack_methods() -> list[str]:
    """List all available attack methods.

    Returns:
        Sorted list of attack method names.

    Examples:
        >>> methods = list_attack_methods()
        >>> "textfooler" in methods
        True
        >>> "bert_attack" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_ATTACK_METHODS)


def get_attack_method(name: str) -> AttackMethod:
    """Get AttackMethod enum from string name.

    Args:
        name: Name of the attack method.

    Returns:
        Corresponding AttackMethod enum value.

    Raises:
        ValueError: If name is not a valid attack method.

    Examples:
        >>> get_attack_method("textfooler")
        <AttackMethod.TEXTFOOLER: 'textfooler'>

        >>> get_attack_method("bert_attack")
        <AttackMethod.BERT_ATTACK: 'bert_attack'>

        >>> get_attack_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid attack method: invalid
    """
    if name not in VALID_ATTACK_METHODS:
        msg = f"invalid attack method: {name}"
        raise ValueError(msg)

    return AttackMethod(name)


def list_ood_detection_methods() -> list[str]:
    """List all available OOD detection methods.

    Returns:
        Sorted list of OOD detection method names.

    Examples:
        >>> methods = list_ood_detection_methods()
        >>> "energy" in methods
        True
        >>> "mahalanobis" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_OOD_DETECTION_METHODS)


def get_ood_detection_method(name: str) -> OODDetectionMethod:
    """Get OODDetectionMethod enum from string name.

    Args:
        name: Name of the OOD detection method.

    Returns:
        Corresponding OODDetectionMethod enum value.

    Raises:
        ValueError: If name is not a valid OOD detection method.

    Examples:
        >>> get_ood_detection_method("energy")
        <OODDetectionMethod.ENERGY: 'energy'>

        >>> get_ood_detection_method("mahalanobis")
        <OODDetectionMethod.MAHALANOBIS: 'mahalanobis'>

        >>> get_ood_detection_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid OOD detection method: invalid
    """
    if name not in VALID_OOD_DETECTION_METHODS:
        msg = f"invalid OOD detection method: {name}"
        raise ValueError(msg)

    return OODDetectionMethod(name)


def apply_perturbation(
    text: str,
    config: PerturbationConfig,
) -> str:
    """Apply perturbation to input text based on configuration.

    Args:
        text: Input text to perturb.
        config: Perturbation configuration.

    Returns:
        Perturbed text.

    Raises:
        ValueError: If text is None.
        ValueError: If config is None.
        ValueError: If text is empty.

    Examples:
        >>> config = PerturbationConfig(
        ...     perturbation_type=PerturbationType.DELETION,
        ...     intensity=0.5,
        ...     max_perturbations=1,
        ... )
        >>> # Note: Deletion perturbation removes words
        >>> result = apply_perturbation("hello world", config)
        >>> len(result) <= len("hello world")
        True

        >>> config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        >>> result = apply_perturbation("test", config)
        >>> isinstance(result, str)
        True

        >>> apply_perturbation(None, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> apply_perturbation("", config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be empty
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if not text:
        msg = "text cannot be empty"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_perturbation_config(config)

    # Apply perturbation based on type
    if config.perturbation_type == PerturbationType.TYPO:
        return _apply_typo_perturbation(text, config)
    elif config.perturbation_type == PerturbationType.SYNONYM:
        return _apply_synonym_perturbation(text, config)
    elif config.perturbation_type == PerturbationType.DELETION:
        return _apply_deletion_perturbation(text, config)
    elif config.perturbation_type == PerturbationType.INSERTION:
        return _apply_insertion_perturbation(text, config)
    elif config.perturbation_type == PerturbationType.PARAPHRASE:
        return _apply_paraphrase_perturbation(text, config)

    return text


def _apply_typo_perturbation(text: str, config: PerturbationConfig) -> str:
    """Apply typo-based perturbation (character swaps).

    Args:
        text: Input text.
        config: Perturbation configuration.

    Returns:
        Text with typos introduced.
    """
    if len(text) < 2:
        return text

    chars = list(text)
    num_swaps = min(
        config.max_perturbations,
        max(1, int(len(chars) * config.intensity)),
    )

    # Find valid swap positions (not at start/end, not spaces)
    valid_positions = [
        i for i in range(1, len(chars) - 1) if chars[i] != " " and chars[i - 1] != " "
    ]

    if not valid_positions:
        return text

    # Use deterministic selection for reproducibility in tests
    positions = valid_positions[: min(num_swaps, len(valid_positions))]

    for pos in positions:
        if pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    return "".join(chars)


def _apply_synonym_perturbation(text: str, config: PerturbationConfig) -> str:
    """Apply synonym-based perturbation.

    Args:
        text: Input text.
        config: Perturbation configuration.

    Returns:
        Text with synonym replacements.
    """
    # Simple synonym mapping for demonstration
    synonyms = {
        "good": "excellent",
        "bad": "poor",
        "big": "large",
        "small": "tiny",
        "fast": "quick",
        "slow": "sluggish",
        "happy": "joyful",
        "sad": "unhappy",
    }

    words = text.split()
    num_replacements = min(
        config.max_perturbations,
        max(1, int(len(words) * config.intensity)),
    )

    replaced = 0
    for i, word in enumerate(words):
        if replaced >= num_replacements:
            break
        lower_word = word.lower()
        if lower_word in synonyms:
            # Preserve case
            if word.isupper():
                words[i] = synonyms[lower_word].upper()
            elif word[0].isupper():
                words[i] = synonyms[lower_word].capitalize()
            else:
                words[i] = synonyms[lower_word]
            replaced += 1

    return " ".join(words)


def _apply_deletion_perturbation(text: str, config: PerturbationConfig) -> str:
    """Apply word deletion perturbation.

    Args:
        text: Input text.
        config: Perturbation configuration.

    Returns:
        Text with words deleted.
    """
    words = text.split()
    if len(words) <= 1:
        return text

    num_deletions = min(
        config.max_perturbations,
        max(1, int(len(words) * config.intensity)),
        len(words) - 1,  # Keep at least one word
    )

    # Delete from the end for determinism
    result_words = words[: len(words) - num_deletions]
    return " ".join(result_words)


def _apply_insertion_perturbation(text: str, config: PerturbationConfig) -> str:
    """Apply word insertion perturbation.

    Args:
        text: Input text.
        config: Perturbation configuration.

    Returns:
        Text with words inserted.
    """
    filler_words = ["very", "really", "quite", "somewhat", "rather"]

    words = text.split()
    num_insertions = min(
        config.max_perturbations,
        max(1, int(len(words) * config.intensity)),
    )

    # Insert at deterministic positions
    for i in range(num_insertions):
        if i < len(words):
            insert_pos = min(i + 1, len(words))
            filler = filler_words[i % len(filler_words)]
            words.insert(insert_pos, filler)

    return " ".join(words)


def _apply_paraphrase_perturbation(text: str, config: PerturbationConfig) -> str:
    """Apply paraphrase-style perturbation.

    Args:
        text: Input text.
        config: Perturbation configuration.

    Returns:
        Paraphrased text.
    """
    # Simple paraphrase patterns
    patterns = [
        ("is", "seems to be"),
        ("are", "appear to be"),
        ("was", "had been"),
        ("can", "is able to"),
        ("will", "is going to"),
    ]

    result = text
    num_replacements = min(
        config.max_perturbations,
        max(1, int(len(text.split()) * config.intensity)),
    )

    replaced = 0
    for old, new in patterns:
        if replaced >= num_replacements:
            break
        if f" {old} " in result:
            result = result.replace(f" {old} ", f" {new} ", 1)
            replaced += 1

    return result


def calculate_robustness_score(
    original_accuracy: float,
    perturbed_accuracy: float,
) -> float:
    """Calculate robustness score from accuracy degradation.

    The robustness score measures how well a model maintains
    performance under perturbations. A score of 1.0 indicates
    perfect robustness (no degradation).

    Args:
        original_accuracy: Model accuracy on clean data.
        perturbed_accuracy: Model accuracy on perturbed data.

    Returns:
        Robustness score between 0 and 1.

    Raises:
        ValueError: If original_accuracy is not in [0, 1].
        ValueError: If perturbed_accuracy is not in [0, 1].

    Examples:
        >>> calculate_robustness_score(0.9, 0.81)
        0.9
        >>> calculate_robustness_score(0.8, 0.8)
        1.0
        >>> calculate_robustness_score(1.0, 0.5)
        0.5

        >>> calculate_robustness_score(1.5, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_accuracy must be between 0 and 1
    """
    if not 0.0 <= original_accuracy <= 1.0:
        msg = f"original_accuracy must be between 0 and 1, got {original_accuracy}"
        raise ValueError(msg)

    if not 0.0 <= perturbed_accuracy <= 1.0:
        msg = f"perturbed_accuracy must be between 0 and 1, got {perturbed_accuracy}"
        raise ValueError(msg)

    if original_accuracy == 0.0:
        return 0.0 if perturbed_accuracy == 0.0 else 1.0

    return min(perturbed_accuracy / original_accuracy, 1.0)


def detect_ood_samples(
    scores: Sequence[float],
    config: OODConfig,
) -> list[bool]:
    """Detect out-of-distribution samples based on scores.

    Args:
        scores: Sequence of OOD detection scores for each sample.
        config: OOD detection configuration.

    Returns:
        List of boolean values indicating OOD status (True = OOD).

    Raises:
        ValueError: If scores is None.
        ValueError: If scores is empty.
        ValueError: If config is None.

    Examples:
        >>> config = OODConfig(
        ...     detection_method=OODDetectionMethod.ENERGY,
        ...     threshold=0.5,
        ... )
        >>> scores = [0.3, 0.6, 0.4, 0.8]
        >>> detect_ood_samples(scores, config)
        [False, True, False, True]

        >>> detect_ood_samples(None, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores cannot be None

        >>> detect_ood_samples([], config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores cannot be empty
    """
    if scores is None:
        msg = "scores cannot be None"
        raise ValueError(msg)

    if len(scores) == 0:
        msg = "scores cannot be empty"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_ood_config(config)

    # Apply temperature scaling if not MSP
    if config.detection_method != OODDetectionMethod.MSP:
        scaled_scores = [s / config.temperature for s in scores]
    else:
        scaled_scores = list(scores)

    # Threshold comparison (higher score = more likely OOD)
    return [score > config.threshold for score in scaled_scores]


def calculate_attack_success_rate(
    original_predictions: Sequence[int],
    attacked_predictions: Sequence[int],
    original_labels: Sequence[int],
) -> float:
    """Calculate the success rate of adversarial attacks.

    An attack is successful if it causes a correct prediction
    to become incorrect.

    Args:
        original_predictions: Predictions on clean data.
        attacked_predictions: Predictions on adversarial data.
        original_labels: Ground truth labels.

    Returns:
        Attack success rate as a float between 0 and 1.

    Raises:
        ValueError: If any input is None.
        ValueError: If inputs have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> original = [1, 0, 1, 1]
        >>> attacked = [0, 0, 0, 1]
        >>> labels = [1, 0, 1, 1]
        >>> calculate_attack_success_rate(original, attacked, labels)
        0.5

        >>> calculate_attack_success_rate(
        ...     None, [1], [1]
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_predictions cannot be None
    """
    if original_predictions is None:
        msg = "original_predictions cannot be None"
        raise ValueError(msg)

    if attacked_predictions is None:
        msg = "attacked_predictions cannot be None"
        raise ValueError(msg)

    if original_labels is None:
        msg = "original_labels cannot be None"
        raise ValueError(msg)

    if len(original_predictions) == 0:
        msg = "inputs cannot be empty"
        raise ValueError(msg)

    if not (
        len(original_predictions) == len(attacked_predictions) == len(original_labels)
    ):
        msg = (
            f"inputs must have same length, got "
            f"{len(original_predictions)}, {len(attacked_predictions)}, "
            f"{len(original_labels)}"
        )
        raise ValueError(msg)

    # Count originally correct predictions
    originally_correct = 0
    successful_attacks = 0

    for orig, attacked, label in zip(
        original_predictions,
        attacked_predictions,
        original_labels,
        strict=True,
    ):
        if orig == label:
            originally_correct += 1
            if attacked != label:
                successful_attacks += 1

    if originally_correct == 0:
        return 0.0

    return successful_attacks / originally_correct


def format_robustness_result(result: RobustnessResult) -> str:
    """Format a robustness result as a human-readable string.

    Args:
        result: Robustness result to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> result = RobustnessResult(
        ...     accuracy_under_perturbation=0.85,
        ...     attack_success_rate=0.15,
        ...     ood_detection_auroc=0.92,
        ... )
        >>> formatted = format_robustness_result(result)
        >>> "Accuracy under Perturbation" in formatted
        True
        >>> "85.00%" in formatted
        True

        >>> format_robustness_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    lines = [
        "Robustness Evaluation Results",
        "=" * 30,
        f"Accuracy under Perturbation: {result.accuracy_under_perturbation:.2%}",
        f"Attack Success Rate: {result.attack_success_rate:.2%}",
        f"OOD Detection AUROC: {result.ood_detection_auroc:.4f}",
    ]

    # Add interpretation
    lines.append("")
    lines.append("Interpretation:")

    if result.accuracy_under_perturbation >= 0.8:
        lines.append("  - Good perturbation robustness")
    elif result.accuracy_under_perturbation >= 0.6:
        lines.append("  - Moderate perturbation robustness")
    else:
        lines.append("  - Low perturbation robustness")

    if result.attack_success_rate <= 0.2:
        lines.append("  - Good adversarial robustness")
    elif result.attack_success_rate <= 0.4:
        lines.append("  - Moderate adversarial robustness")
    else:
        lines.append("  - Low adversarial robustness")

    if result.ood_detection_auroc >= 0.9:
        lines.append("  - Excellent OOD detection")
    elif result.ood_detection_auroc >= 0.7:
        lines.append("  - Good OOD detection")
    else:
        lines.append("  - Poor OOD detection")

    return "\n".join(lines)


def get_recommended_robustness_config(
    model_type: str,
) -> dict[str, Any]:
    """Get recommended robustness testing configuration for a model type.

    Args:
        model_type: Type of model (e.g., "text_classification", "sentiment",
            "nli", "qa").

    Returns:
        Dictionary with recommended configurations.

    Raises:
        ValueError: If model_type is None or empty.

    Examples:
        >>> config = get_recommended_robustness_config("text_classification")
        >>> "perturbation" in config
        True
        >>> "adversarial" in config
        True
        >>> "ood" in config
        True

        >>> get_recommended_robustness_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty

        >>> get_recommended_robustness_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be None
    """
    if model_type is None:
        msg = "model_type cannot be None"
        raise ValueError(msg)

    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    # Default configuration
    base_config: dict[str, Any] = {
        "perturbation": {
            "type": PerturbationType.TYPO,
            "intensity": 0.1,
            "max_perturbations": 5,
        },
        "adversarial": {
            "method": AttackMethod.TEXTFOOLER,
            "max_queries": 100,
            "success_threshold": 0.5,
        },
        "ood": {
            "method": OODDetectionMethod.ENERGY,
            "threshold": 0.5,
            "calibration_data_size": 1000,
        },
    }

    # Model-specific configurations
    model_configs: dict[str, dict[str, Any]] = {
        "text_classification": {
            "perturbation": {
                "type": PerturbationType.SYNONYM,
                "intensity": 0.2,
                "max_perturbations": 3,
            },
            "adversarial": {
                "method": AttackMethod.TEXTFOOLER,
                "max_queries": 200,
                "success_threshold": 0.5,
            },
            "ood": {
                "method": OODDetectionMethod.MSP,
                "threshold": 0.6,
                "calibration_data_size": 500,
            },
        },
        "sentiment": {
            "perturbation": {
                "type": PerturbationType.SYNONYM,
                "intensity": 0.15,
                "max_perturbations": 4,
            },
            "adversarial": {
                "method": AttackMethod.BERT_ATTACK,
                "max_queries": 150,
                "success_threshold": 0.4,
            },
            "ood": {
                "method": OODDetectionMethod.ENTROPY,
                "threshold": 0.5,
                "calibration_data_size": 800,
            },
        },
        "nli": {
            "perturbation": {
                "type": PerturbationType.PARAPHRASE,
                "intensity": 0.25,
                "max_perturbations": 2,
            },
            "adversarial": {
                "method": AttackMethod.PWWS,
                "max_queries": 300,
                "success_threshold": 0.6,
            },
            "ood": {
                "method": OODDetectionMethod.MAHALANOBIS,
                "threshold": 0.7,
                "calibration_data_size": 1000,
            },
        },
        "qa": {
            "perturbation": {
                "type": PerturbationType.TYPO,
                "intensity": 0.1,
                "max_perturbations": 5,
            },
            "adversarial": {
                "method": AttackMethod.DEEPWORDBUG,
                "max_queries": 100,
                "success_threshold": 0.3,
            },
            "ood": {
                "method": OODDetectionMethod.ENERGY,
                "threshold": 0.4,
                "calibration_data_size": 1200,
            },
        },
    }

    # Return model-specific config if available, otherwise base config
    return model_configs.get(model_type.lower(), base_config)
