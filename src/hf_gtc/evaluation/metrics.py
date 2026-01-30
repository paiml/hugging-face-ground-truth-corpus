"""Evaluation metrics for ML models.

This module provides functions for computing common evaluation metrics
used in machine learning tasks, including BLEU, ROUGE, BERTScore,
perplexity, and classification metrics.

Examples:
    >>> from hf_gtc.evaluation.metrics import MetricType, RougeVariant
    >>> MetricType.BLEU.value
    'bleu'
    >>> RougeVariant.ROUGE1.value
    'rouge1'
    >>> from hf_gtc.evaluation.metrics import compute_accuracy
    >>> compute_accuracy([1, 0, 1], [1, 0, 0])
    0.6666666666666666
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class MetricType(Enum):
    """Types of evaluation metrics.

    Attributes:
        BLEU: Bilingual Evaluation Understudy score.
        ROUGE: Recall-Oriented Understudy for Gisting Evaluation.
        BERTSCORE: Contextual embedding-based similarity score.
        METEOR: Metric for Evaluation of Translation with Explicit ORdering.
        PERPLEXITY: Perplexity score for language models.
        ACCURACY: Classification accuracy.
        F1: F1 score (harmonic mean of precision and recall).
        EXACT_MATCH: Exact string match.

    Examples:
        >>> MetricType.BLEU.value
        'bleu'
        >>> MetricType.ROUGE.value
        'rouge'
        >>> MetricType.BERTSCORE.value
        'bertscore'
        >>> MetricType.PERPLEXITY.value
        'perplexity'
    """

    BLEU = "bleu"
    ROUGE = "rouge"
    BERTSCORE = "bertscore"
    METEOR = "meteor"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    F1 = "f1"
    EXACT_MATCH = "exact_match"


VALID_METRIC_TYPES = frozenset(m.value for m in MetricType)


class RougeVariant(Enum):
    """Variants of ROUGE metric.

    Attributes:
        ROUGE1: Unigram overlap.
        ROUGE2: Bigram overlap.
        ROUGEL: Longest common subsequence.
        ROUGELSUM: LCS over sentences.

    Examples:
        >>> RougeVariant.ROUGE1.value
        'rouge1'
        >>> RougeVariant.ROUGE2.value
        'rouge2'
        >>> RougeVariant.ROUGEL.value
        'rougeL'
        >>> RougeVariant.ROUGELSUM.value
        'rougeLsum'
    """

    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    ROUGELSUM = "rougeLsum"


VALID_ROUGE_VARIANTS = frozenset(v.value for v in RougeVariant)


class AggregationMethod(Enum):
    """Methods for aggregating scores across samples.

    Attributes:
        MICRO: Aggregate at the token/element level.
        MACRO: Average per-sample scores equally.
        WEIGHTED: Weight by sample importance/frequency.

    Examples:
        >>> AggregationMethod.MICRO.value
        'micro'
        >>> AggregationMethod.MACRO.value
        'macro'
        >>> AggregationMethod.WEIGHTED.value
        'weighted'
    """

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"


VALID_AGGREGATION_METHODS = frozenset(a.value for a in AggregationMethod)


@dataclass(frozen=True, slots=True)
class BLEUConfig:
    """Configuration for BLEU score computation.

    Attributes:
        max_ngram: Maximum n-gram order to consider. Defaults to 4.
        smoothing: Smoothing method (none, add_k, floor). Defaults to add_k.
        tokenizer: Tokenizer type (word, char, subword). Defaults to word.

    Examples:
        >>> config = BLEUConfig()
        >>> config.max_ngram
        4
        >>> config.smoothing
        'add_k'
        >>> config = BLEUConfig(max_ngram=2, smoothing="floor")
        >>> config.max_ngram
        2
    """

    max_ngram: int = 4
    smoothing: str = "add_k"
    tokenizer: str = "word"


@dataclass(frozen=True, slots=True)
class ROUGEConfig:
    """Configuration for ROUGE score computation.

    Attributes:
        variants: List of ROUGE variants to compute.
        use_stemmer: Whether to apply stemming. Defaults to False.
        split_summaries: Whether to split into sentences. Defaults to True.

    Examples:
        >>> config = ROUGEConfig()
        >>> len(config.variants) > 0
        True
        >>> config.use_stemmer
        False
        >>> config = ROUGEConfig(use_stemmer=True, split_summaries=False)
        >>> config.use_stemmer
        True
    """

    variants: tuple[RougeVariant, ...] = (RougeVariant.ROUGE1, RougeVariant.ROUGE2)
    use_stemmer: bool = False
    split_summaries: bool = True


@dataclass(frozen=True, slots=True)
class BERTScoreConfig:
    """Configuration for BERTScore computation.

    Attributes:
        model_name: Name of the model to use. Defaults to microsoft/deberta-xlarge-mnli.
        num_layers: Number of layers to use. Defaults to None (use all).
        rescale_with_baseline: Whether to rescale scores. Defaults to True.

    Examples:
        >>> config = BERTScoreConfig()
        >>> config.model_name
        'microsoft/deberta-xlarge-mnli'
        >>> config.rescale_with_baseline
        True
        >>> config = BERTScoreConfig(model_name="bert-base-uncased", num_layers=8)
        >>> config.num_layers
        8
    """

    model_name: str = "microsoft/deberta-xlarge-mnli"
    num_layers: int | None = None
    rescale_with_baseline: bool = True


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Configuration for metric computation.

    Attributes:
        metric_type: Type of metric to compute.
        bleu_config: BLEU-specific configuration.
        rouge_config: ROUGE-specific configuration.
        bertscore_config: BERTScore-specific configuration.

    Examples:
        >>> config = MetricConfig(metric_type=MetricType.BLEU)
        >>> config.metric_type
        <MetricType.BLEU: 'bleu'>
        >>> config = MetricConfig(metric_type=MetricType.BLEU, bleu_config=BLEUConfig())
        >>> config.bleu_config is not None
        True
    """

    metric_type: MetricType
    bleu_config: BLEUConfig | None = None
    rouge_config: ROUGEConfig | None = None
    bertscore_config: BERTScoreConfig | None = None


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Result of metric computation.

    Attributes:
        score: Primary metric score.
        precision: Precision component (if applicable).
        recall: Recall component (if applicable).
        f1: F1 component (if applicable).
        confidence: Confidence interval (if computed).

    Examples:
        >>> result = MetricResult(score=0.85)
        >>> result.score
        0.85
        >>> result.precision is None
        True
        >>> result = MetricResult(score=0.85, precision=0.9, recall=0.8, f1=0.85)
        >>> result.f1
        0.85
    """

    score: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    confidence: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Container for classification metrics.

    Attributes:
        accuracy: Overall accuracy (correct / total).
        precision: Precision score (true positives / predicted positives).
        recall: Recall score (true positives / actual positives).
        f1: F1 score (harmonic mean of precision and recall).

    Examples:
        >>> metrics = ClassificationMetrics(
        ...     accuracy=0.9, precision=0.85, recall=0.88, f1=0.865
        ... )
        >>> metrics.accuracy
        0.9
    """

    accuracy: float
    precision: float
    recall: float
    f1: float


# Factory functions


def create_bleu_config(
    max_ngram: int = 4,
    smoothing: str = "add_k",
    tokenizer: str = "word",
) -> BLEUConfig:
    """Create a BLEU configuration.

    Args:
        max_ngram: Maximum n-gram order. Defaults to 4.
        smoothing: Smoothing method (none, add_k, floor). Defaults to add_k.
        tokenizer: Tokenizer type (word, char, subword). Defaults to word.

    Returns:
        Configured BLEUConfig instance.

    Examples:
        >>> config = create_bleu_config()
        >>> config.max_ngram
        4
        >>> config = create_bleu_config(max_ngram=2, smoothing="floor")
        >>> config.max_ngram
        2
        >>> config.smoothing
        'floor'
    """
    return BLEUConfig(
        max_ngram=max_ngram,
        smoothing=smoothing,
        tokenizer=tokenizer,
    )


def create_rouge_config(
    variants: tuple[RougeVariant, ...] | None = None,
    use_stemmer: bool = False,
    split_summaries: bool = True,
) -> ROUGEConfig:
    """Create a ROUGE configuration.

    Args:
        variants: ROUGE variants to compute. Defaults to (ROUGE1, ROUGE2).
        use_stemmer: Whether to apply stemming. Defaults to False.
        split_summaries: Whether to split into sentences. Defaults to True.

    Returns:
        Configured ROUGEConfig instance.

    Examples:
        >>> config = create_rouge_config()
        >>> RougeVariant.ROUGE1 in config.variants
        True
        >>> config = create_rouge_config(
        ...     variants=(RougeVariant.ROUGEL,), use_stemmer=True
        ... )
        >>> config.use_stemmer
        True
    """
    if variants is None:
        variants = (RougeVariant.ROUGE1, RougeVariant.ROUGE2)
    return ROUGEConfig(
        variants=variants,
        use_stemmer=use_stemmer,
        split_summaries=split_summaries,
    )


def create_bertscore_config(
    model_name: str = "microsoft/deberta-xlarge-mnli",
    num_layers: int | None = None,
    rescale_with_baseline: bool = True,
) -> BERTScoreConfig:
    """Create a BERTScore configuration.

    Args:
        model_name: Name of the model. Defaults to microsoft/deberta-xlarge-mnli.
        num_layers: Number of layers to use. Defaults to None.
        rescale_with_baseline: Whether to rescale. Defaults to True.

    Returns:
        Configured BERTScoreConfig instance.

    Examples:
        >>> config = create_bertscore_config()
        >>> config.model_name
        'microsoft/deberta-xlarge-mnli'
        >>> config = create_bertscore_config(model_name="bert-base-uncased")
        >>> config.model_name
        'bert-base-uncased'
    """
    return BERTScoreConfig(
        model_name=model_name,
        num_layers=num_layers,
        rescale_with_baseline=rescale_with_baseline,
    )


def create_metric_config(
    metric_type: MetricType,
    bleu_config: BLEUConfig | None = None,
    rouge_config: ROUGEConfig | None = None,
    bertscore_config: BERTScoreConfig | None = None,
) -> MetricConfig:
    """Create a metric configuration.

    Automatically creates default sub-configs based on metric type.

    Args:
        metric_type: Type of metric to compute.
        bleu_config: BLEU-specific configuration.
        rouge_config: ROUGE-specific configuration.
        bertscore_config: BERTScore-specific configuration.

    Returns:
        Configured MetricConfig instance.

    Examples:
        >>> config = create_metric_config(MetricType.BLEU)
        >>> config.metric_type
        <MetricType.BLEU: 'bleu'>
        >>> config.bleu_config is not None
        True
        >>> config = create_metric_config(MetricType.ROUGE)
        >>> config.rouge_config is not None
        True
    """
    # Auto-create sub-configs based on metric type
    if metric_type == MetricType.BLEU and bleu_config is None:
        bleu_config = create_bleu_config()
    elif metric_type == MetricType.ROUGE and rouge_config is None:
        rouge_config = create_rouge_config()
    elif metric_type == MetricType.BERTSCORE and bertscore_config is None:
        bertscore_config = create_bertscore_config()

    return MetricConfig(
        metric_type=metric_type,
        bleu_config=bleu_config,
        rouge_config=rouge_config,
        bertscore_config=bertscore_config,
    )


# Validation functions


def validate_bleu_config(config: BLEUConfig) -> None:
    """Validate BLEU configuration.

    Args:
        config: BLEUConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_ngram is not positive.
        ValueError: If smoothing is not valid.
        ValueError: If tokenizer is empty.

    Examples:
        >>> config = create_bleu_config()
        >>> validate_bleu_config(config)  # No error

        >>> validate_bleu_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BLEUConfig(max_ngram=0)
        >>> validate_bleu_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_ngram must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.max_ngram <= 0:
        msg = f"max_ngram must be positive, got {config.max_ngram}"
        raise ValueError(msg)

    valid_smoothing = {"none", "add_k", "floor"}
    if config.smoothing not in valid_smoothing:
        msg = f"smoothing must be one of {valid_smoothing}, got {config.smoothing}"
        raise ValueError(msg)

    if not config.tokenizer:
        msg = "tokenizer cannot be empty"
        raise ValueError(msg)


def validate_rouge_config(config: ROUGEConfig) -> None:
    """Validate ROUGE configuration.

    Args:
        config: ROUGEConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If variants is empty.

    Examples:
        >>> config = create_rouge_config()
        >>> validate_rouge_config(config)  # No error

        >>> validate_rouge_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ROUGEConfig(variants=())
        >>> validate_rouge_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: variants cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.variants:
        msg = "variants cannot be empty"
        raise ValueError(msg)


def validate_bertscore_config(config: BERTScoreConfig) -> None:
    """Validate BERTScore configuration.

    Args:
        config: BERTScoreConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If model_name is empty.
        ValueError: If num_layers is not positive when specified.

    Examples:
        >>> config = create_bertscore_config()
        >>> validate_bertscore_config(config)  # No error

        >>> validate_bertscore_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BERTScoreConfig(model_name="")
        >>> validate_bertscore_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if config.num_layers is not None and config.num_layers <= 0:
        msg = f"num_layers must be positive, got {config.num_layers}"
        raise ValueError(msg)


def validate_metric_config(config: MetricConfig) -> None:
    """Validate metric configuration.

    Args:
        config: MetricConfig to validate.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_metric_config(MetricType.BLEU)
        >>> validate_metric_config(config)  # No error

        >>> validate_metric_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.bleu_config is not None:
        validate_bleu_config(config.bleu_config)
    if config.rouge_config is not None:
        validate_rouge_config(config.rouge_config)
    if config.bertscore_config is not None:
        validate_bertscore_config(config.bertscore_config)


def validate_metric_result(result: MetricResult) -> None:
    """Validate metric result.

    Args:
        result: MetricResult to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If score is NaN.

    Examples:
        >>> result = MetricResult(score=0.85)
        >>> validate_metric_result(result)  # No error

        >>> validate_metric_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    if math.isnan(result.score):
        msg = "score cannot be NaN"
        raise ValueError(msg)


# List/get functions for enums


def list_metric_types() -> list[str]:
    """List all available metric types.

    Returns:
        Sorted list of metric type names.

    Examples:
        >>> types = list_metric_types()
        >>> "bleu" in types
        True
        >>> "rouge" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_METRIC_TYPES)


def validate_metric_type(metric_type: str) -> bool:
    """Validate if a string is a valid metric type.

    Args:
        metric_type: The metric type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_metric_type("bleu")
        True
        >>> validate_metric_type("rouge")
        True
        >>> validate_metric_type("invalid")
        False
        >>> validate_metric_type("")
        False
    """
    return metric_type in VALID_METRIC_TYPES


def get_metric_type(name: str) -> MetricType:
    """Get MetricType enum from string name.

    Args:
        name: Name of the metric type.

    Returns:
        Corresponding MetricType enum value.

    Raises:
        ValueError: If name is not a valid metric type.

    Examples:
        >>> get_metric_type("bleu")
        <MetricType.BLEU: 'bleu'>
        >>> get_metric_type("rouge")
        <MetricType.ROUGE: 'rouge'>

        >>> get_metric_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid metric type: invalid
    """
    if not validate_metric_type(name):
        msg = f"invalid metric type: {name}"
        raise ValueError(msg)

    return MetricType(name)


def list_rouge_variants() -> list[str]:
    """List all available ROUGE variants.

    Returns:
        Sorted list of ROUGE variant names.

    Examples:
        >>> variants = list_rouge_variants()
        >>> "rouge1" in variants
        True
        >>> "rougeL" in variants
        True
    """
    return sorted(VALID_ROUGE_VARIANTS)


def validate_rouge_variant(variant: str) -> bool:
    """Validate if a string is a valid ROUGE variant.

    Args:
        variant: The variant string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_rouge_variant("rouge1")
        True
        >>> validate_rouge_variant("rougeL")
        True
        >>> validate_rouge_variant("invalid")
        False
        >>> validate_rouge_variant("")
        False
    """
    return variant in VALID_ROUGE_VARIANTS


def get_rouge_variant(name: str) -> RougeVariant:
    """Get RougeVariant enum from string name.

    Args:
        name: Name of the ROUGE variant.

    Returns:
        Corresponding RougeVariant enum value.

    Raises:
        ValueError: If name is not a valid ROUGE variant.

    Examples:
        >>> get_rouge_variant("rouge1")
        <RougeVariant.ROUGE1: 'rouge1'>
        >>> get_rouge_variant("rougeL")
        <RougeVariant.ROUGEL: 'rougeL'>

        >>> get_rouge_variant("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid ROUGE variant: invalid
    """
    if not validate_rouge_variant(name):
        msg = f"invalid ROUGE variant: {name}"
        raise ValueError(msg)

    return RougeVariant(name)


def list_aggregation_methods() -> list[str]:
    """List all available aggregation methods.

    Returns:
        Sorted list of aggregation method names.

    Examples:
        >>> methods = list_aggregation_methods()
        >>> "micro" in methods
        True
        >>> "macro" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_AGGREGATION_METHODS)


def validate_aggregation_method(method: str) -> bool:
    """Validate if a string is a valid aggregation method.

    Args:
        method: The method string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_aggregation_method("micro")
        True
        >>> validate_aggregation_method("macro")
        True
        >>> validate_aggregation_method("invalid")
        False
        >>> validate_aggregation_method("")
        False
    """
    return method in VALID_AGGREGATION_METHODS


def get_aggregation_method(name: str) -> AggregationMethod:
    """Get AggregationMethod enum from string name.

    Args:
        name: Name of the aggregation method.

    Returns:
        Corresponding AggregationMethod enum value.

    Raises:
        ValueError: If name is not a valid aggregation method.

    Examples:
        >>> get_aggregation_method("micro")
        <AggregationMethod.MICRO: 'micro'>
        >>> get_aggregation_method("macro")
        <AggregationMethod.MACRO: 'macro'>

        >>> get_aggregation_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid aggregation method: invalid
    """
    if not validate_aggregation_method(name):
        msg = f"invalid aggregation method: {name}"
        raise ValueError(msg)

    return AggregationMethod(name)


# Core metric calculation functions


def _tokenize(text: str, method: str = "word") -> list[str]:
    """Tokenize text using the specified method.

    Args:
        text: Text to tokenize.
        method: Tokenization method (word, char).

    Returns:
        List of tokens.

    Examples:
        >>> _tokenize("hello world")
        ['hello', 'world']
        >>> _tokenize("hello", "char")
        ['h', 'e', 'l', 'l', 'o']
    """
    if method == "char":
        return list(text)
    return text.lower().split()


def _get_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    """Extract n-grams from tokens.

    Args:
        tokens: List of tokens.
        n: N-gram size.

    Returns:
        Dictionary of n-grams to counts.

    Examples:
        >>> ngrams = _get_ngrams(["a", "b", "c"], 2)
        >>> ngrams[("a", "b")]
        1
    """
    ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams


def _compute_ngram_precision(
    candidate_ngrams: dict[tuple[str, ...], int],
    reference_ngrams: dict[tuple[str, ...], int],
) -> tuple[int, int]:
    """Compute clipped n-gram precision.

    Args:
        candidate_ngrams: Candidate n-gram counts.
        reference_ngrams: Reference n-gram counts.

    Returns:
        Tuple of (clipped matches, total candidate n-grams).
    """
    matches = 0
    total = sum(candidate_ngrams.values())

    for ngram, count in candidate_ngrams.items():
        ref_count = reference_ngrams.get(ngram, 0)
        matches += min(count, ref_count)

    return matches, total


def calculate_bleu(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    config: BLEUConfig | None = None,
) -> MetricResult:
    """Calculate BLEU score.

    Args:
        candidates: Candidate translations.
        references: Reference translations (multiple per candidate).
        config: BLEU configuration. Uses defaults if None.

    Returns:
        MetricResult with BLEU score.

    Raises:
        ValueError: If candidates is None or empty.
        ValueError: If references is None or empty.
        ValueError: If candidates and references have different lengths.

    Examples:
        >>> cands = ["the cat sat on the mat"]
        >>> refs = [["the cat sat on the mat"]]
        >>> result = calculate_bleu(cands, refs)
        >>> result.score > 0.9
        True

        >>> cands = ["the dog jumped over the fence"]
        >>> refs = [["the cat sat on the mat"]]
        >>> result = calculate_bleu(cands, refs)
        >>> result.score < 0.5
        True

        >>> calculate_bleu([], [[]])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: candidates cannot be empty
    """
    if candidates is None:
        msg = "candidates cannot be None"
        raise ValueError(msg)

    if len(candidates) == 0:
        msg = "candidates cannot be empty"
        raise ValueError(msg)

    if references is None:
        msg = "references cannot be None"
        raise ValueError(msg)

    if len(references) == 0:
        msg = "references cannot be empty"
        raise ValueError(msg)

    if len(candidates) != len(references):
        msg = (
            f"candidates and references must have the same length, "
            f"got {len(candidates)} and {len(references)}"
        )
        raise ValueError(msg)

    if config is None:
        config = create_bleu_config()

    # Accumulate n-gram statistics
    total_matches = [0] * config.max_ngram
    total_counts = [0] * config.max_ngram
    total_cand_length = 0
    total_ref_length = 0

    for cand, refs in zip(candidates, references, strict=True):
        cand_tokens = _tokenize(cand, config.tokenizer)
        total_cand_length += len(cand_tokens)

        # Find closest reference length
        ref_lengths = [len(_tokenize(ref, config.tokenizer)) for ref in refs]
        closest_ref_length = min(
            ref_lengths, key=lambda x: (abs(x - len(cand_tokens)), x)
        )
        total_ref_length += closest_ref_length

        # Compute n-gram matches
        for n in range(1, config.max_ngram + 1):
            cand_ngrams = _get_ngrams(cand_tokens, n)

            # Merge reference n-grams (take max count for each)
            merged_ref_ngrams: dict[tuple[str, ...], int] = {}
            for ref in refs:
                ref_tokens = _tokenize(ref, config.tokenizer)
                ref_ngrams = _get_ngrams(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    merged_ref_ngrams[ngram] = max(
                        merged_ref_ngrams.get(ngram, 0), count
                    )

            matches, total = _compute_ngram_precision(cand_ngrams, merged_ref_ngrams)
            total_matches[n - 1] += matches
            total_counts[n - 1] += total

    # Compute modified precision for each n-gram
    precisions = []
    for n in range(config.max_ngram):
        if total_counts[n] == 0:
            if config.smoothing == "add_k":
                precisions.append(1.0 / (total_counts[n] + 1))
            elif config.smoothing == "floor":
                precisions.append(0.1)
            else:
                precisions.append(0.0)
        else:
            if config.smoothing == "add_k" and total_matches[n] == 0:
                precisions.append(1.0 / (total_counts[n] + 1))
            else:
                precisions.append(total_matches[n] / total_counts[n])

    # Compute brevity penalty
    if total_cand_length <= total_ref_length and total_cand_length > 0:
        bp = math.exp(1 - total_ref_length / total_cand_length)
    else:
        bp = 1.0

    # Compute geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_sum = sum(math.log(p) for p in precisions) / config.max_ngram
        geo_mean = math.exp(log_sum)
    else:
        geo_mean = 0.0

    bleu_score = bp * geo_mean

    return MetricResult(score=bleu_score)


def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
    """Compute length of longest common subsequence.

    Args:
        seq1: First sequence.
        seq2: Second sequence.

    Returns:
        Length of LCS.
    """
    m, n = len(seq1), len(seq2)
    # Use space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def calculate_rouge(
    candidates: Sequence[str],
    references: Sequence[str],
    config: ROUGEConfig | None = None,
) -> dict[str, MetricResult]:
    """Calculate ROUGE scores.

    Args:
        candidates: Candidate summaries.
        references: Reference summaries.
        config: ROUGE configuration. Uses defaults if None.

    Returns:
        Dictionary mapping ROUGE variant to MetricResult.

    Raises:
        ValueError: If candidates is None or empty.
        ValueError: If references is None or empty.
        ValueError: If candidates and references have different lengths.

    Examples:
        >>> cands = ["the cat sat on the mat"]
        >>> refs = ["the cat was on the mat"]
        >>> results = calculate_rouge(cands, refs)
        >>> "rouge1" in results
        True
        >>> results["rouge1"].score > 0.5
        True

        >>> calculate_rouge([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: candidates cannot be empty
    """
    if candidates is None:
        msg = "candidates cannot be None"
        raise ValueError(msg)

    if len(candidates) == 0:
        msg = "candidates cannot be empty"
        raise ValueError(msg)

    if references is None:
        msg = "references cannot be None"
        raise ValueError(msg)

    if len(references) == 0:
        msg = "references cannot be empty"
        raise ValueError(msg)

    if len(candidates) != len(references):
        msg = (
            f"candidates and references must have the same length, "
            f"got {len(candidates)} and {len(references)}"
        )
        raise ValueError(msg)

    if config is None:
        config = create_rouge_config()

    results: dict[str, MetricResult] = {}

    for variant in config.variants:
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for cand, ref in zip(candidates, references, strict=True):
            cand_tokens = _tokenize(cand)
            ref_tokens = _tokenize(ref)

            if variant == RougeVariant.ROUGE1:
                # Unigram overlap
                cand_set = set(cand_tokens)
                ref_set = set(ref_tokens)
                overlap = len(cand_set & ref_set)
                precision = overlap / len(cand_set) if cand_set else 0.0
                recall = overlap / len(ref_set) if ref_set else 0.0
            elif variant == RougeVariant.ROUGE2:
                # Bigram overlap
                cand_ngrams = _get_ngrams(cand_tokens, 2)
                ref_ngrams = _get_ngrams(ref_tokens, 2)
                cand_set = set(cand_ngrams.keys())
                ref_set = set(ref_ngrams.keys())
                overlap = len(cand_set & ref_set)
                precision = overlap / len(cand_set) if cand_set else 0.0
                recall = overlap / len(ref_set) if ref_set else 0.0
            elif variant in (RougeVariant.ROUGEL, RougeVariant.ROUGELSUM):
                # LCS-based
                lcs_len = _lcs_length(cand_tokens, ref_tokens)
                precision = lcs_len / len(cand_tokens) if cand_tokens else 0.0
                recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
            else:
                precision = 0.0
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        n = len(candidates)
        avg_precision = total_precision / n
        avg_recall = total_recall / n
        avg_f1 = total_f1 / n

        results[variant.value] = MetricResult(
            score=avg_f1,
            precision=avg_precision,
            recall=avg_recall,
            f1=avg_f1,
        )

    return results


def calculate_bertscore(
    candidates: Sequence[str],
    references: Sequence[str],
    config: BERTScoreConfig | None = None,
) -> MetricResult:
    """Calculate BERTScore.

    Note: This is a simplified implementation that computes word overlap
    as a proxy. For production use, integrate with the actual BERTScore library.

    Args:
        candidates: Candidate texts.
        references: Reference texts.
        config: BERTScore configuration. Uses defaults if None.

    Returns:
        MetricResult with BERTScore (precision, recall, F1).

    Raises:
        ValueError: If candidates is None or empty.
        ValueError: If references is None or empty.
        ValueError: If candidates and references have different lengths.

    Examples:
        >>> cands = ["the cat sat on the mat"]
        >>> refs = ["the cat was sitting on the mat"]
        >>> result = calculate_bertscore(cands, refs)
        >>> result.score > 0.3
        True
        >>> result.precision is not None
        True

        >>> calculate_bertscore([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: candidates cannot be empty
    """
    if candidates is None:
        msg = "candidates cannot be None"
        raise ValueError(msg)

    if len(candidates) == 0:
        msg = "candidates cannot be empty"
        raise ValueError(msg)

    if references is None:
        msg = "references cannot be None"
        raise ValueError(msg)

    if len(references) == 0:
        msg = "references cannot be empty"
        raise ValueError(msg)

    if len(candidates) != len(references):
        msg = (
            f"candidates and references must have the same length, "
            f"got {len(candidates)} and {len(references)}"
        )
        raise ValueError(msg)

    if config is None:
        config = create_bertscore_config()

    # Simplified BERTScore proxy: word overlap with soft matching
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for cand, ref in zip(candidates, references, strict=True):
        cand_tokens = set(_tokenize(cand))
        ref_tokens = set(_tokenize(ref))

        if not cand_tokens or not ref_tokens:
            continue

        # Simple token matching (proxy for embedding similarity)
        overlap = len(cand_tokens & ref_tokens)
        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    n = len(candidates)
    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_f1 = total_f1 / n

    # Apply rescaling if configured
    if config.rescale_with_baseline:
        # Apply mild rescaling (baseline subtraction)
        baseline = 0.3
        avg_precision = max(0.0, (avg_precision - baseline) / (1.0 - baseline))
        avg_recall = max(0.0, (avg_recall - baseline) / (1.0 - baseline))
        if avg_precision + avg_recall > 0:
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            avg_f1 = 0.0

    return MetricResult(
        score=avg_f1,
        precision=avg_precision,
        recall=avg_recall,
        f1=avg_f1,
    )


def calculate_perplexity(
    log_probs: Sequence[float] | None = None,
    loss: float | None = None,
) -> MetricResult:
    """Calculate perplexity from log probabilities or loss.

    Perplexity = exp(average negative log probability) or exp(loss)

    Args:
        log_probs: Sequence of log probabilities (base e).
        loss: Cross-entropy loss value (alternative to log_probs).

    Returns:
        MetricResult with perplexity score.

    Raises:
        ValueError: If neither log_probs nor loss is provided.
        ValueError: If loss is negative.
        ValueError: If loss would cause overflow.

    Examples:
        >>> result = calculate_perplexity(loss=2.0)
        >>> 7.38 < result.score < 7.40
        True

        >>> result = calculate_perplexity(log_probs=[-1.0, -2.0, -1.5])
        >>> result.score > 1.0
        True

        >>> calculate_perplexity()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: either log_probs or loss must be provided

        >>> calculate_perplexity(loss=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: loss cannot be negative
    """
    if log_probs is None and loss is None:
        msg = "either log_probs or loss must be provided"
        raise ValueError(msg)

    if loss is not None:
        if loss < 0:
            msg = f"loss cannot be negative, got {loss}"
            raise ValueError(msg)

        if loss > 700:
            msg = f"loss too large, would cause overflow: {loss}"
            raise ValueError(msg)

        perplexity = math.exp(loss)
    else:
        # Compute from log_probs
        assert log_probs is not None
        if len(log_probs) == 0:
            msg = "log_probs cannot be empty"
            raise ValueError(msg)

        avg_neg_log_prob = -sum(log_probs) / len(log_probs)

        if avg_neg_log_prob > 700:
            msg = "perplexity too large, would cause overflow"
            raise ValueError(msg)

        perplexity = math.exp(avg_neg_log_prob)

    return MetricResult(score=perplexity)


def aggregate_metrics(
    results: Sequence[MetricResult],
    method: AggregationMethod = AggregationMethod.MACRO,
    weights: Sequence[float] | None = None,
) -> MetricResult:
    """Aggregate multiple metric results.

    Args:
        results: Sequence of MetricResult objects.
        method: Aggregation method. Defaults to macro.
        weights: Weights for weighted aggregation.

    Returns:
        Aggregated MetricResult.

    Raises:
        ValueError: If results is None or empty.
        ValueError: If weights provided but different length from results.

    Examples:
        >>> r1 = MetricResult(score=0.8, precision=0.9, recall=0.7, f1=0.8)
        >>> r2 = MetricResult(score=0.9, precision=0.85, recall=0.95, f1=0.9)
        >>> agg = aggregate_metrics([r1, r2])
        >>> 0.84 < agg.score < 0.86
        True

        >>> aggregate_metrics([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be empty
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    if len(results) == 0:
        msg = "results cannot be empty"
        raise ValueError(msg)

    if weights is not None and len(weights) != len(results):
        msg = (
            f"weights must have same length as results, "
            f"got {len(weights)} and {len(results)}"
        )
        raise ValueError(msg)

    if method == AggregationMethod.WEIGHTED and weights is None:
        # Default to uniform weights
        weights = [1.0 / len(results)] * len(results)
    elif method == AggregationMethod.MACRO:
        weights = [1.0 / len(results)] * len(results)
    elif method == AggregationMethod.MICRO:
        # For micro, we'd need token counts which we don't have
        # Fall back to macro
        weights = [1.0 / len(results)] * len(results)

    # Normalize weights
    assert weights is not None  # guaranteed by branches above
    total_weight = float(sum(weights))
    if total_weight > 0:
        weights = [w / total_weight for w in weights]

    # Aggregate scores
    agg_score = float(sum(r.score * w for r, w in zip(results, weights, strict=True)))

    # Aggregate precision, recall, f1 if available
    agg_precision: float | None = None
    agg_recall: float | None = None
    agg_f1: float | None = None

    if all(r.precision is not None for r in results):
        agg_precision = float(
            sum(float(r.precision) * w for r, w in zip(results, weights, strict=True))
        )

    if all(r.recall is not None for r in results):
        agg_recall = float(
            sum(float(r.recall) * w for r, w in zip(results, weights, strict=True))
        )

    if all(r.f1 is not None for r in results):
        agg_f1 = float(
            sum(float(r.f1) * w for r, w in zip(results, weights, strict=True))
        )

    return MetricResult(
        score=agg_score,
        precision=agg_precision,
        recall=agg_recall,
        f1=agg_f1,
    )


def format_metric_stats(
    results: dict[str, MetricResult],
) -> str:
    """Format metric results as a human-readable string.

    Args:
        results: Dictionary mapping metric names to results.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If results is None.

    Examples:
        >>> results = {
        ...     "bleu": MetricResult(score=0.45),
        ...     "rouge1": MetricResult(score=0.65, precision=0.7, recall=0.6, f1=0.65),
        ... }
        >>> formatted = format_metric_stats(results)
        >>> "bleu" in formatted.lower()
        True
        >>> "rouge1" in formatted.lower()
        True

        >>> format_metric_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be None
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    lines = ["Metric Results", "=" * 40]

    for name, result in sorted(results.items()):
        lines.append(f"\n{name.upper()}:")
        lines.append(f"  Score: {result.score:.4f}")
        if result.precision is not None:
            lines.append(f"  Precision: {result.precision:.4f}")
        if result.recall is not None:
            lines.append(f"  Recall: {result.recall:.4f}")
        if result.f1 is not None:
            lines.append(f"  F1: {result.f1:.4f}")
        if result.confidence is not None:
            lo, hi = result.confidence
            lines.append(f"  Confidence: [{lo:.4f}, {hi:.4f}]")

    return "\n".join(lines)


def get_recommended_metric_config(
    task: str,
) -> MetricConfig:
    """Get recommended metric configuration for a task.

    Args:
        task: Task type (translation, summarization, qa, generation).

    Returns:
        Recommended MetricConfig.

    Raises:
        ValueError: If task is empty.

    Examples:
        >>> config = get_recommended_metric_config("translation")
        >>> config.metric_type
        <MetricType.BLEU: 'bleu'>

        >>> config = get_recommended_metric_config("summarization")
        >>> config.metric_type
        <MetricType.ROUGE: 'rouge'>

        >>> get_recommended_metric_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task cannot be empty
    """
    if not task:
        msg = "task cannot be empty"
        raise ValueError(msg)

    task_lower = task.lower()

    if task_lower in ("translation", "mt", "machine_translation"):
        return create_metric_config(
            MetricType.BLEU,
            bleu_config=BLEUConfig(max_ngram=4, smoothing="add_k", tokenizer="word"),
        )
    elif task_lower in ("summarization", "summary", "abstractive"):
        return create_metric_config(
            MetricType.ROUGE,
            rouge_config=ROUGEConfig(
                variants=(
                    RougeVariant.ROUGE1,
                    RougeVariant.ROUGE2,
                    RougeVariant.ROUGEL,
                ),
                use_stemmer=True,
                split_summaries=True,
            ),
        )
    elif task_lower in ("qa", "question_answering"):
        return create_metric_config(MetricType.EXACT_MATCH)
    elif task_lower in ("generation", "text_generation", "lm"):
        return create_metric_config(MetricType.PERPLEXITY)
    elif task_lower in ("similarity", "semantic", "nli"):
        return create_metric_config(
            MetricType.BERTSCORE,
            bertscore_config=BERTScoreConfig(
                model_name="microsoft/deberta-xlarge-mnli",
                rescale_with_baseline=True,
            ),
        )
    else:
        # Default to BLEU
        return create_metric_config(MetricType.BLEU)


# Existing classification metrics (preserved from original module)


def compute_accuracy(
    predictions: Sequence[int],
    labels: Sequence[int],
) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.

    Returns:
        Accuracy as a float between 0 and 1.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_accuracy([1, 0, 1, 1], [1, 0, 1, 0])
        0.75
        >>> compute_accuracy([1, 1, 1], [1, 1, 1])
        1.0
        >>> compute_accuracy([0, 0], [1, 1])
        0.0

        >>> compute_accuracy([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions and labels cannot be empty

        >>> compute_accuracy([1], [1, 2])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions and labels must have the same length
    """
    if len(predictions) == 0 or len(labels) == 0:
        msg = "predictions and labels cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(labels):
        msg = (
            f"predictions and labels must have the same length, "
            f"got {len(predictions)} and {len(labels)}"
        )
        raise ValueError(msg)

    correct = sum(pred == lbl for pred, lbl in zip(predictions, labels, strict=True))
    return correct / len(predictions)


def compute_precision(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> float:
    """Compute precision for binary classification.

    Precision = True Positives / (True Positives + False Positives)

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        Precision as a float between 0 and 1.
        Returns 0.0 if no positive predictions were made.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_precision([1, 1, 0, 1], [1, 0, 0, 1])
        0.6666666666666666
        >>> compute_precision([0, 0, 0], [1, 1, 1])
        0.0
        >>> compute_precision([1, 1], [1, 1])
        1.0
    """
    if len(predictions) == 0 or len(labels) == 0:
        msg = "predictions and labels cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(labels):
        msg = (
            f"predictions and labels must have the same length, "
            f"got {len(predictions)} and {len(labels)}"
        )
        raise ValueError(msg)

    true_positives = sum(
        pred == positive_label and lbl == positive_label
        for pred, lbl in zip(predictions, labels, strict=True)
    )
    predicted_positives = sum(p == positive_label for p in predictions)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def compute_recall(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> float:
    """Compute recall for binary classification.

    Recall = True Positives / (True Positives + False Negatives)

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        Recall as a float between 0 and 1.
        Returns 0.0 if no actual positives exist.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_recall([1, 1, 0, 1], [1, 0, 0, 1])
        1.0
        >>> compute_recall([0, 0, 0], [1, 1, 1])
        0.0
        >>> compute_recall([1, 0], [1, 1])
        0.5
    """
    if len(predictions) == 0 or len(labels) == 0:
        msg = "predictions and labels cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(labels):
        msg = (
            f"predictions and labels must have the same length, "
            f"got {len(predictions)} and {len(labels)}"
        )
        raise ValueError(msg)

    true_positives = sum(
        pred == positive_label and lbl == positive_label
        for pred, lbl in zip(predictions, labels, strict=True)
    )
    actual_positives = sum(lbl == positive_label for lbl in labels)

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def compute_f1(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> float:
    """Compute F1 score for binary classification.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        F1 score as a float between 0 and 1.
        Returns 0.0 if precision + recall is 0.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_f1([1, 1, 0, 1], [1, 0, 0, 1])
        0.8
        >>> compute_f1([0, 0, 0], [1, 1, 1])
        0.0
        >>> compute_f1([1, 1], [1, 1])
        1.0
    """
    precision = compute_precision(predictions, labels, positive_label)
    recall = compute_recall(predictions, labels, positive_label)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_classification_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> ClassificationMetrics:
    """Compute all classification metrics at once.

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        ClassificationMetrics containing accuracy, precision, recall, and F1.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> metrics = compute_classification_metrics([1, 1, 0, 1], [1, 0, 0, 1])
        >>> metrics.accuracy
        0.75
        >>> round(metrics.precision, 4)
        0.6667
        >>> metrics.recall
        1.0
        >>> metrics.f1
        0.8
    """
    return ClassificationMetrics(
        accuracy=compute_accuracy(predictions, labels),
        precision=compute_precision(predictions, labels, positive_label),
        recall=compute_recall(predictions, labels, positive_label),
        f1=compute_f1(predictions, labels, positive_label),
    )


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity score.

    Raises:
        ValueError: If loss is negative.
        ValueError: If loss would cause overflow.

    Examples:
        >>> round(compute_perplexity(2.0), 4)
        7.3891
        >>> round(compute_perplexity(0.0), 4)
        1.0
        >>> round(compute_perplexity(1.0), 4)
        2.7183

        >>> compute_perplexity(-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: loss cannot be negative
    """
    if loss < 0:
        msg = f"loss cannot be negative, got {loss}"
        raise ValueError(msg)

    # Prevent overflow for very large losses
    if loss > 700:
        msg = f"loss too large, would cause overflow: {loss}"
        raise ValueError(msg)

    return math.exp(loss)


def compute_mean_loss(losses: Sequence[float]) -> float:
    """Compute mean loss from a sequence of losses.

    Args:
        losses: Sequence of loss values.

    Returns:
        Mean loss.

    Raises:
        ValueError: If losses is empty.

    Examples:
        >>> compute_mean_loss([1.0, 2.0, 3.0])
        2.0
        >>> compute_mean_loss([0.5])
        0.5

        >>> compute_mean_loss([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: losses cannot be empty
    """
    if len(losses) == 0:
        msg = "losses cannot be empty"
        raise ValueError(msg)

    return sum(losses) / len(losses)


def create_compute_metrics_fn(
    positive_label: int = 1,
) -> Callable[[tuple[Any, Any]], dict[str, float]]:
    """Create a compute_metrics function for HuggingFace Trainer.

    Args:
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        A function compatible with Trainer.compute_metrics.

    Examples:
        >>> fn = create_compute_metrics_fn()
        >>> callable(fn)
        True
    """

    def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
        """Compute metrics from evaluation predictions.

        Args:
            eval_pred: Tuple of (predictions, labels) from Trainer.

        Returns:
            Dictionary of metric names to values.
        """
        predictions, labels = eval_pred

        # Handle logits (take argmax)
        if len(predictions.shape) > 1:
            predictions = predictions.argmax(axis=-1)

        preds_list = predictions.tolist()
        labels_list = labels.tolist()

        metrics = compute_classification_metrics(
            preds_list, labels_list, positive_label
        )

        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        }

    return compute_metrics
