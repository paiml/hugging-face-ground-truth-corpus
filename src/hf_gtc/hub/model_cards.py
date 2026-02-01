"""Model card generation and documentation utilities for HuggingFace Hub.

This module provides functions for generating, validating, and managing model cards
following HuggingFace Hub best practices.

Examples:
    >>> from hf_gtc.hub.model_cards import create_model_metadata, generate_model_card
    >>> metadata = create_model_metadata(name="bert-base")
    >>> metadata.name
    'bert-base'
    >>> len(generate_model_card(create_model_card_config(metadata=metadata))) > 0
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class CardSection(Enum):
    """Model card sections.

    Attributes:
        OVERVIEW: Model overview and summary.
        USAGE: How to use the model.
        TRAINING: Training details and procedure.
        EVALUATION: Evaluation results and benchmarks.
        LIMITATIONS: Known limitations and biases.
        ETHICS: Ethical considerations and guidelines.

    Examples:
        >>> CardSection.OVERVIEW.value
        'overview'
        >>> CardSection.TRAINING.value
        'training'
        >>> CardSection.ETHICS.value
        'ethics'
    """

    OVERVIEW = "overview"
    USAGE = "usage"
    TRAINING = "training"
    EVALUATION = "evaluation"
    LIMITATIONS = "limitations"
    ETHICS = "ethics"


VALID_CARD_SECTIONS = frozenset(s.value for s in CardSection)


class LicenseType(Enum):
    """License types for models.

    Attributes:
        MIT: MIT License.
        APACHE2: Apache License 2.0.
        CC_BY: Creative Commons Attribution.
        CC_BY_NC: Creative Commons Attribution Non-Commercial.
        PROPRIETARY: Proprietary license.
        OPENRAIL: OpenRAIL license.

    Examples:
        >>> LicenseType.MIT.value
        'mit'
        >>> LicenseType.APACHE2.value
        'apache2'
        >>> LicenseType.OPENRAIL.value
        'openrail'
    """

    MIT = "mit"
    APACHE2 = "apache2"
    CC_BY = "cc_by"
    CC_BY_NC = "cc_by_nc"
    PROPRIETARY = "proprietary"
    OPENRAIL = "openrail"


VALID_LICENSE_TYPES = frozenset(lt.value for lt in LicenseType)


class ModelTask(Enum):
    """Model task types.

    Attributes:
        TEXT_CLASSIFICATION: Text classification task.
        TEXT_GENERATION: Text generation task.
        QUESTION_ANSWERING: Question answering task.
        SUMMARIZATION: Text summarization task.
        TRANSLATION: Language translation task.

    Examples:
        >>> ModelTask.TEXT_CLASSIFICATION.value
        'text_classification'
        >>> ModelTask.TEXT_GENERATION.value
        'text_generation'
        >>> ModelTask.TRANSLATION.value
        'translation'
    """

    TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


VALID_MODEL_TASKS = frozenset(t.value for t in ModelTask)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata for a model card.

    Attributes:
        name: Model name.
        version: Model version.
        author: Model author or organization.
        license: License type.
        tags: List of tags.
        pipeline_tag: Pipeline task tag.

    Examples:
        >>> metadata = ModelMetadata(
        ...     name="bert-base",
        ...     version="1.0.0",
        ...     author="huggingface",
        ...     license=LicenseType.APACHE2,
        ...     tags=("nlp", "bert"),
        ...     pipeline_tag=ModelTask.TEXT_CLASSIFICATION,
        ... )
        >>> metadata.name
        'bert-base'
        >>> metadata.license
        <LicenseType.APACHE2: 'apache2'>
    """

    name: str
    version: str
    author: str
    license: LicenseType
    tags: tuple[str, ...]
    pipeline_tag: ModelTask | None


@dataclass(frozen=True, slots=True)
class TrainingDetails:
    """Training details for a model.

    Attributes:
        dataset: Training dataset name or identifier.
        epochs: Number of training epochs.
        batch_size: Batch size used during training.
        learning_rate: Learning rate.
        hardware: Hardware used for training.

    Examples:
        >>> details = TrainingDetails(
        ...     dataset="wikipedia",
        ...     epochs=3,
        ...     batch_size=32,
        ...     learning_rate=2e-5,
        ...     hardware="8x A100 GPUs",
        ... )
        >>> details.epochs
        3
        >>> details.learning_rate
        2e-05
    """

    dataset: str
    epochs: int
    batch_size: int
    learning_rate: float
    hardware: str


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    """Evaluation results for a model.

    Attributes:
        metrics: List of metric names.
        datasets: List of evaluation datasets.
        scores: Dictionary mapping metric names to scores.

    Examples:
        >>> results = EvaluationResults(
        ...     metrics=("accuracy", "f1"),
        ...     datasets=("glue", "superglue"),
        ...     scores={"accuracy": 0.92, "f1": 0.89},
        ... )
        >>> results.scores["accuracy"]
        0.92
        >>> "f1" in results.metrics
        True
    """

    metrics: tuple[str, ...]
    datasets: tuple[str, ...]
    scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class ModelCardConfig:
    """Configuration for model card generation.

    Attributes:
        metadata: Model metadata.
        training: Training details (optional).
        evaluation: Evaluation results (optional).
        sections: Sections to include in the card.
        language: Model language(s).

    Examples:
        >>> metadata = ModelMetadata(
        ...     "bert", "1.0", "hf", LicenseType.MIT, (), None
        ... )
        >>> config = ModelCardConfig(
        ...     metadata=metadata,
        ...     training=None,
        ...     evaluation=None,
        ...     sections=(CardSection.OVERVIEW, CardSection.USAGE),
        ...     language=("en",),
        ... )
        >>> config.language
        ('en',)
    """

    metadata: ModelMetadata
    training: TrainingDetails | None
    evaluation: EvaluationResults | None
    sections: tuple[CardSection, ...]
    language: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelCardStats:
    """Statistics about a generated model card.

    Attributes:
        completeness_score: Score from 0.0 to 1.0 indicating completeness.
        missing_sections: List of missing recommended sections.
        word_count: Total word count of the card.

    Examples:
        >>> stats = ModelCardStats(
        ...     completeness_score=0.85,
        ...     missing_sections=("ethics",),
        ...     word_count=500,
        ... )
        >>> stats.completeness_score
        0.85
        >>> "ethics" in stats.missing_sections
        True
    """

    completeness_score: float
    missing_sections: tuple[str, ...]
    word_count: int


def validate_model_metadata(metadata: ModelMetadata) -> None:
    """Validate model metadata.

    Args:
        metadata: Model metadata to validate.

    Raises:
        ValueError: If metadata is invalid.

    Examples:
        >>> metadata = ModelMetadata(
        ...     "bert", "1.0", "hf", LicenseType.MIT, (), None
        ... )
        >>> validate_model_metadata(metadata)  # No error

        >>> bad = ModelMetadata("", "1.0", "hf", LicenseType.MIT, (), None)
        >>> validate_model_metadata(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if not metadata.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if not metadata.version:
        msg = "version cannot be empty"
        raise ValueError(msg)

    if not metadata.author:
        msg = "author cannot be empty"
        raise ValueError(msg)


def validate_training_details(details: TrainingDetails) -> None:
    """Validate training details.

    Args:
        details: Training details to validate.

    Raises:
        ValueError: If details are invalid.

    Examples:
        >>> details = TrainingDetails("dataset", 3, 32, 2e-5, "GPU")
        >>> validate_training_details(details)  # No error

        >>> bad = TrainingDetails("", 3, 32, 2e-5, "GPU")
        >>> validate_training_details(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset cannot be empty
    """
    if not details.dataset:
        msg = "dataset cannot be empty"
        raise ValueError(msg)

    if details.epochs <= 0:
        msg = f"epochs must be positive, got {details.epochs}"
        raise ValueError(msg)

    if details.batch_size <= 0:
        msg = f"batch_size must be positive, got {details.batch_size}"
        raise ValueError(msg)

    if details.learning_rate <= 0:
        msg = f"learning_rate must be positive, got {details.learning_rate}"
        raise ValueError(msg)

    if not details.hardware:
        msg = "hardware cannot be empty"
        raise ValueError(msg)


def validate_evaluation_results(results: EvaluationResults) -> None:
    """Validate evaluation results.

    Args:
        results: Evaluation results to validate.

    Raises:
        ValueError: If results are invalid.

    Examples:
        >>> results = EvaluationResults(
        ...     ("accuracy",), ("glue",), {"accuracy": 0.9}
        ... )
        >>> validate_evaluation_results(results)  # No error

        >>> bad = EvaluationResults((), ("glue",), {})
        >>> validate_evaluation_results(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metrics cannot be empty
    """
    if not results.metrics:
        msg = "metrics cannot be empty"
        raise ValueError(msg)

    if not results.datasets:
        msg = "datasets cannot be empty"
        raise ValueError(msg)

    _validate_metrics_in_scores(results.metrics, results.scores)
    _validate_scores_numeric(results.scores)


def _validate_metrics_in_scores(
    metrics: tuple[str, ...], scores: dict[str, int | float]
) -> None:
    """Validate all metrics have corresponding scores."""
    for metric in metrics:
        if metric not in scores:
            msg = f"metric '{metric}' not found in scores"
            raise ValueError(msg)


def _validate_scores_numeric(scores: dict[str, int | float]) -> None:
    """Validate all scores are numeric."""
    for metric, score in scores.items():
        if not isinstance(score, (int, float)):
            msg = f"score for '{metric}' must be numeric, got {type(score).__name__}"
            raise ValueError(msg)


def validate_model_card_config(config: ModelCardConfig) -> None:
    """Validate model card configuration.

    Args:
        config: Model card configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> metadata = ModelMetadata(
        ...     "bert", "1.0", "hf", LicenseType.MIT, (), None
        ... )
        >>> config = ModelCardConfig(metadata, None, None, (), ("en",))
        >>> validate_model_card_config(config)  # No error

        >>> bad_meta = ModelMetadata("", "1.0", "hf", LicenseType.MIT, (), None)
        >>> bad_config = ModelCardConfig(bad_meta, None, None, (), ("en",))
        >>> validate_model_card_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    validate_model_metadata(config.metadata)

    if config.training is not None:
        validate_training_details(config.training)

    if config.evaluation is not None:
        validate_evaluation_results(config.evaluation)

    if not config.language:
        msg = "language cannot be empty"
        raise ValueError(msg)

    for lang in config.language:
        if not lang:
            msg = "language entries cannot be empty strings"
            raise ValueError(msg)


def create_model_metadata(
    name: str,
    version: str = "1.0.0",
    author: str = "unknown",
    license_type: str = "mit",
    tags: tuple[str, ...] | list[str] | None = None,
    pipeline_tag: str | None = None,
) -> ModelMetadata:
    """Create model metadata.

    Args:
        name: Model name.
        version: Model version. Defaults to "1.0.0".
        author: Model author. Defaults to "unknown".
        license_type: License type. Defaults to "mit".
        tags: Model tags. Defaults to None.
        pipeline_tag: Pipeline task tag. Defaults to None.

    Returns:
        ModelMetadata with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> metadata = create_model_metadata("bert-base")
        >>> metadata.name
        'bert-base'
        >>> metadata.version
        '1.0.0'

        >>> metadata = create_model_metadata(
        ...     "gpt2",
        ...     version="2.0.0",
        ...     license_type="apache2",
        ...     tags=["nlp", "generation"],
        ...     pipeline_tag="text_generation",
        ... )
        >>> metadata.license
        <LicenseType.APACHE2: 'apache2'>
        >>> metadata.tags
        ('nlp', 'generation')

        >>> create_model_metadata("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty

        >>> create_model_metadata(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "x", license_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: license_type must be one of
    """
    if license_type not in VALID_LICENSE_TYPES:
        msg = f"license_type must be one of {VALID_LICENSE_TYPES}, got '{license_type}'"
        raise ValueError(msg)

    if pipeline_tag is not None and pipeline_tag not in VALID_MODEL_TASKS:
        msg = f"pipeline_tag must be one of {VALID_MODEL_TASKS}, got '{pipeline_tag}'"
        raise ValueError(msg)

    tags_tuple = tuple(tags) if tags is not None else ()
    pipeline = ModelTask(pipeline_tag) if pipeline_tag is not None else None

    metadata = ModelMetadata(
        name=name,
        version=version,
        author=author,
        license=LicenseType(license_type),
        tags=tags_tuple,
        pipeline_tag=pipeline,
    )
    validate_model_metadata(metadata)
    return metadata


def create_training_details(
    dataset: str,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    hardware: str = "GPU",
) -> TrainingDetails:
    """Create training details.

    Args:
        dataset: Training dataset name.
        epochs: Number of training epochs. Defaults to 3.
        batch_size: Training batch size. Defaults to 32.
        learning_rate: Learning rate. Defaults to 2e-5.
        hardware: Hardware used. Defaults to "GPU".

    Returns:
        TrainingDetails with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> details = create_training_details("wikipedia")
        >>> details.dataset
        'wikipedia'
        >>> details.epochs
        3

        >>> details = create_training_details(
        ...     "imdb",
        ...     epochs=5,
        ...     batch_size=64,
        ...     learning_rate=1e-4,
        ...     hardware="8x A100",
        ... )
        >>> details.batch_size
        64
        >>> details.hardware
        '8x A100'

        >>> create_training_details("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset cannot be empty

        >>> create_training_details("x", epochs=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: epochs must be positive
    """
    details = TrainingDetails(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hardware=hardware,
    )
    validate_training_details(details)
    return details


def create_evaluation_results(
    metrics: tuple[str, ...] | list[str],
    datasets: tuple[str, ...] | list[str],
    scores: dict[str, float],
) -> EvaluationResults:
    """Create evaluation results.

    Args:
        metrics: List of metric names.
        datasets: List of evaluation datasets.
        scores: Dictionary of metric scores.

    Returns:
        EvaluationResults with the specified data.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> results = create_evaluation_results(
        ...     ["accuracy", "f1"],
        ...     ["glue"],
        ...     {"accuracy": 0.92, "f1": 0.89},
        ... )
        >>> results.scores["accuracy"]
        0.92

        >>> create_evaluation_results(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     [], ["glue"], {}
        ... )
        Traceback (most recent call last):
        ValueError: metrics cannot be empty

        >>> create_evaluation_results(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     ["acc"], ["glue"], {}
        ... )
        Traceback (most recent call last):
        ValueError: metric 'acc' not found in scores
    """
    metrics_tuple = tuple(metrics)
    datasets_tuple = tuple(datasets)

    results = EvaluationResults(
        metrics=metrics_tuple,
        datasets=datasets_tuple,
        scores=scores,
    )
    validate_evaluation_results(results)
    return results


def create_model_card_config(
    metadata: ModelMetadata,
    training: TrainingDetails | None = None,
    evaluation: EvaluationResults | None = None,
    sections: tuple[CardSection, ...] | list[CardSection] | None = None,
    language: tuple[str, ...] | list[str] | None = None,
) -> ModelCardConfig:
    """Create model card configuration.

    Args:
        metadata: Model metadata.
        training: Training details. Defaults to None.
        evaluation: Evaluation results. Defaults to None.
        sections: Sections to include. Defaults to all sections.
        language: Model language(s). Defaults to ("en",).

    Returns:
        ModelCardConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> metadata = create_model_metadata("bert")
        >>> config = create_model_card_config(metadata)
        >>> config.language
        ('en',)
        >>> len(config.sections) > 0
        True

        >>> config = create_model_card_config(
        ...     metadata,
        ...     sections=[CardSection.OVERVIEW, CardSection.USAGE],
        ...     language=["en", "de"],
        ... )
        >>> config.language
        ('en', 'de')
    """
    sections_tuple = tuple(CardSection) if sections is None else tuple(sections)

    if language is None:
        language_tuple: tuple[str, ...] = ("en",)
    else:
        language_tuple = tuple(language)

    config = ModelCardConfig(
        metadata=metadata,
        training=training,
        evaluation=evaluation,
        sections=sections_tuple,
        language=language_tuple,
    )
    validate_model_card_config(config)
    return config


def list_card_sections() -> list[str]:
    """List all available card sections.

    Returns:
        Sorted list of section names.

    Examples:
        >>> sections = list_card_sections()
        >>> "overview" in sections
        True
        >>> "training" in sections
        True
        >>> sections == sorted(sections)
        True
    """
    return sorted(VALID_CARD_SECTIONS)


def list_license_types() -> list[str]:
    """List all available license types.

    Returns:
        Sorted list of license type names.

    Examples:
        >>> licenses = list_license_types()
        >>> "mit" in licenses
        True
        >>> "apache2" in licenses
        True
        >>> licenses == sorted(licenses)
        True
    """
    return sorted(VALID_LICENSE_TYPES)


def list_model_tasks() -> list[str]:
    """List all available model tasks.

    Returns:
        Sorted list of model task names.

    Examples:
        >>> tasks = list_model_tasks()
        >>> "text_classification" in tasks
        True
        >>> "text_generation" in tasks
        True
        >>> tasks == sorted(tasks)
        True
    """
    return sorted(VALID_MODEL_TASKS)


def get_card_section(name: str) -> CardSection:
    """Get card section from string name.

    Args:
        name: Section name.

    Returns:
        CardSection enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_card_section("overview")
        <CardSection.OVERVIEW: 'overview'>

        >>> get_card_section("training")
        <CardSection.TRAINING: 'training'>

        >>> get_card_section("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: card_section must be one of
    """
    if name not in VALID_CARD_SECTIONS:
        msg = f"card_section must be one of {VALID_CARD_SECTIONS}, got '{name}'"
        raise ValueError(msg)
    return CardSection(name)


def get_license_type(name: str) -> LicenseType:
    """Get license type from string name.

    Args:
        name: License type name.

    Returns:
        LicenseType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_license_type("mit")
        <LicenseType.MIT: 'mit'>

        >>> get_license_type("apache2")
        <LicenseType.APACHE2: 'apache2'>

        >>> get_license_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: license_type must be one of
    """
    if name not in VALID_LICENSE_TYPES:
        msg = f"license_type must be one of {VALID_LICENSE_TYPES}, got '{name}'"
        raise ValueError(msg)
    return LicenseType(name)


def get_model_task(name: str) -> ModelTask:
    """Get model task from string name.

    Args:
        name: Model task name.

    Returns:
        ModelTask enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_model_task("text_classification")
        <ModelTask.TEXT_CLASSIFICATION: 'text_classification'>

        >>> get_model_task("text_generation")
        <ModelTask.TEXT_GENERATION: 'text_generation'>

        >>> get_model_task("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_task must be one of
    """
    if name not in VALID_MODEL_TASKS:
        msg = f"model_task must be one of {VALID_MODEL_TASKS}, got '{name}'"
        raise ValueError(msg)
    return ModelTask(name)


def _generate_frontmatter(config: ModelCardConfig) -> str:
    """Generate YAML frontmatter for a model card.

    Args:
        config: Model card configuration.

    Returns:
        YAML frontmatter string.

    Examples:
        >>> metadata = create_model_metadata("bert", tags=["nlp"])
        >>> config = create_model_card_config(metadata)
        >>> frontmatter = _generate_frontmatter(config)
        >>> "license: mit" in frontmatter
        True
        >>> "---" in frontmatter
        True
    """
    lines = ["---"]
    lines.append(f"license: {config.metadata.license.value}")

    _append_language_frontmatter(lines, config.language)
    _append_tags_frontmatter(lines, config.metadata.tags)

    if config.metadata.pipeline_tag is not None:
        lines.append(f"pipeline_tag: {config.metadata.pipeline_tag.value}")

    lines.append("---")
    return "\n".join(lines)


def _append_language_frontmatter(lines: list[str], language: tuple[str, ...]) -> None:
    """Append language frontmatter entries."""
    if not language:
        return
    if len(language) == 1:
        lines.append(f"language: {language[0]}")
    else:
        lines.append("language:")
        for lang in language:
            lines.append(f"  - {lang}")


def _append_tags_frontmatter(lines: list[str], tags: tuple[str, ...]) -> None:
    """Append tags frontmatter entries."""
    if not tags:
        return
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")


def _generate_overview_section(config: ModelCardConfig) -> str:
    """Generate overview section content.

    Args:
        config: Model card configuration.

    Returns:
        Overview section markdown.

    Examples:
        >>> metadata = create_model_metadata("bert", author="huggingface")
        >>> config = create_model_card_config(metadata)
        >>> overview = _generate_overview_section(config)
        >>> "bert" in overview
        True
    """
    lines = [
        f"# {config.metadata.name}",
        "",
        f"**Version:** {config.metadata.version}",
        f"**Author:** {config.metadata.author}",
        f"**License:** {config.metadata.license.value}",
    ]

    if config.metadata.pipeline_tag is not None:
        lines.append(f"**Task:** {config.metadata.pipeline_tag.value}")

    return "\n".join(lines)


def _generate_usage_section(config: ModelCardConfig) -> str:
    """Generate usage section content.

    Args:
        config: Model card configuration.

    Returns:
        Usage section markdown.

    Examples:
        >>> metadata = create_model_metadata(
        ...     "bert", pipeline_tag="text_classification"
        ... )
        >>> config = create_model_card_config(metadata)
        >>> usage = _generate_usage_section(config)
        >>> "## Usage" in usage
        True
    """
    lines = ["## Usage", ""]

    task = config.metadata.pipeline_tag
    if task is not None:
        lines.append(f"This model can be used for {task.value.replace('_', ' ')}.")
        lines.append("")
        lines.append("```python")
        lines.append("from transformers import pipeline")
        lines.append("")
        lines.append(f'pipe = pipeline("{task.value.replace("_", "-")}")')
        lines.append(f'pipe.model = "{config.metadata.name}"')
        lines.append("```")
    else:
        lines.append("```python")
        lines.append("from transformers import AutoModel")
        lines.append("")
        lines.append(f'model = AutoModel.from_pretrained("{config.metadata.name}")')
        lines.append("```")

    return "\n".join(lines)


def _generate_training_section(config: ModelCardConfig) -> str:
    """Generate training section content.

    Args:
        config: Model card configuration.

    Returns:
        Training section markdown.

    Examples:
        >>> metadata = create_model_metadata("bert")
        >>> training = create_training_details("wikipedia", epochs=5)
        >>> config = create_model_card_config(metadata, training=training)
        >>> section = _generate_training_section(config)
        >>> "wikipedia" in section
        True
    """
    lines = ["## Training", ""]

    if config.training is not None:
        lines.append(f"**Dataset:** {config.training.dataset}")
        lines.append(f"**Epochs:** {config.training.epochs}")
        lines.append(f"**Batch Size:** {config.training.batch_size}")
        lines.append(f"**Learning Rate:** {config.training.learning_rate}")
        lines.append(f"**Hardware:** {config.training.hardware}")
    else:
        lines.append("Training details not available.")

    return "\n".join(lines)


def _generate_evaluation_section(config: ModelCardConfig) -> str:
    """Generate evaluation section content.

    Args:
        config: Model card configuration.

    Returns:
        Evaluation section markdown.

    Examples:
        >>> metadata = create_model_metadata("bert")
        >>> evaluation = create_evaluation_results(
        ...     ["accuracy"], ["glue"], {"accuracy": 0.92}
        ... )
        >>> config = create_model_card_config(metadata, evaluation=evaluation)
        >>> section = _generate_evaluation_section(config)
        >>> "0.92" in section
        True
    """
    lines = ["## Evaluation", ""]

    if config.evaluation is not None:
        lines.append("### Datasets")
        for dataset in config.evaluation.datasets:
            lines.append(f"- {dataset}")
        lines.append("")
        lines.append("### Results")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric in config.evaluation.metrics:
            score = config.evaluation.scores.get(metric, "N/A")
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            lines.append(f"| {metric} | {score_str} |")
    else:
        lines.append("Evaluation results not available.")

    return "\n".join(lines)


def _generate_limitations_section(_config: ModelCardConfig) -> str:
    """Generate limitations section content.

    Args:
        _config: Model card configuration.

    Returns:
        Limitations section markdown.

    Examples:
        >>> metadata = create_model_metadata("bert")
        >>> config = create_model_card_config(metadata)
        >>> section = _generate_limitations_section(config)
        >>> "## Limitations" in section
        True
    """
    lines = [
        "## Limitations",
        "",
        "- This model may exhibit biases present in the training data.",
        "- Performance may vary on out-of-distribution data.",
        "- The model should be validated before use in production.",
    ]
    return "\n".join(lines)


def _generate_ethics_section(_config: ModelCardConfig) -> str:
    """Generate ethics section content.

    Args:
        _config: Model card configuration.

    Returns:
        Ethics section markdown.

    Examples:
        >>> metadata = create_model_metadata("bert")
        >>> config = create_model_card_config(metadata)
        >>> section = _generate_ethics_section(config)
        >>> "## Ethical Considerations" in section
        True
    """
    lines = [
        "## Ethical Considerations",
        "",
        "Users should consider the following when using this model:",
        "",
        "- Ensure fair and unbiased use of model predictions.",
        "- Do not use the model for harmful or malicious purposes.",
        "- Evaluate the model's impact on different demographic groups.",
        "- Comply with applicable laws and regulations.",
    ]
    return "\n".join(lines)


def generate_model_card(config: ModelCardConfig) -> str:
    """Generate a complete model card from configuration.

    Args:
        config: Model card configuration.

    Returns:
        Complete model card as markdown string.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> metadata = create_model_metadata("bert-base", author="huggingface")
        >>> config = create_model_card_config(metadata)
        >>> card = generate_model_card(config)
        >>> "bert-base" in card
        True
        >>> "---" in card
        True

        >>> training = create_training_details("wiki", epochs=3)
        >>> config = create_model_card_config(metadata, training=training)
        >>> card = generate_model_card(config)
        >>> "wiki" in card
        True
    """
    validate_model_card_config(config)

    parts = [_generate_frontmatter(config), ""]

    section_generators = {
        CardSection.OVERVIEW: _generate_overview_section,
        CardSection.USAGE: _generate_usage_section,
        CardSection.TRAINING: _generate_training_section,
        CardSection.EVALUATION: _generate_evaluation_section,
        CardSection.LIMITATIONS: _generate_limitations_section,
        CardSection.ETHICS: _generate_ethics_section,
    }

    for section in config.sections:
        generator = section_generators.get(section)
        if generator:
            parts.append(generator(config))
            parts.append("")

    return "\n".join(parts).strip() + "\n"


def calculate_completeness(config: ModelCardConfig) -> float:
    """Calculate the completeness score of a model card configuration.

    Args:
        config: Model card configuration.

    Returns:
        Completeness score from 0.0 to 1.0.

    Examples:
        >>> metadata = create_model_metadata("bert")
        >>> config = create_model_card_config(metadata)
        >>> score = calculate_completeness(config)
        >>> 0.0 <= score <= 1.0
        True

        >>> training = create_training_details("wiki")
        >>> evaluation = create_evaluation_results(
        ...     ["acc"], ["glue"], {"acc": 0.9}
        ... )
        >>> full_config = create_model_card_config(
        ...     metadata, training=training, evaluation=evaluation
        ... )
        >>> full_score = calculate_completeness(full_config)
        >>> full_score > calculate_completeness(config)
        True
    """
    total_weight = 0.0
    earned_weight = 0.0

    # Metadata completeness (weight: 3)
    total_weight += 3.0
    if config.metadata.name:
        earned_weight += 1.0
    if config.metadata.author != "unknown":
        earned_weight += 1.0
    if config.metadata.pipeline_tag is not None:
        earned_weight += 1.0

    # Tags (weight: 1)
    total_weight += 1.0
    if config.metadata.tags:
        earned_weight += 1.0

    # Training details (weight: 2)
    total_weight += 2.0
    if config.training is not None:
        earned_weight += 2.0

    # Evaluation results (weight: 2)
    total_weight += 2.0
    if config.evaluation is not None:
        earned_weight += 2.0

    # Sections (weight: 1 per section)
    all_sections = set(CardSection)
    included_sections = set(config.sections)
    total_weight += len(all_sections)
    earned_weight += len(included_sections)

    return earned_weight / total_weight if total_weight > 0 else 0.0


def validate_card_structure(card_content: str) -> tuple[bool, list[str]]:
    """Validate the structure of a model card.

    Args:
        card_content: Model card markdown content.

    Returns:
        Tuple of (is_valid, list of issues).

    Raises:
        ValueError: If card_content is empty.

    Examples:
        >>> metadata = create_model_metadata("bert")
        >>> config = create_model_card_config(metadata)
        >>> card = generate_model_card(config)
        >>> is_valid, issues = validate_card_structure(card)
        >>> is_valid
        True

        >>> is_valid, issues = validate_card_structure("# Minimal Card")
        >>> is_valid
        False
        >>> len(issues) > 0
        True

        >>> validate_card_structure("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: card_content cannot be empty
    """
    if not card_content or not card_content.strip():
        msg = "card_content cannot be empty"
        raise ValueError(msg)

    issues: list[str] = []

    # Check for frontmatter
    if not card_content.startswith("---"):
        issues.append("Missing YAML frontmatter")

    # Check for title
    if "\n# " not in card_content and not card_content.startswith("# "):
        issues.append("Missing main title (# heading)")

    # Check for essential sections
    recommended_sections = ["Usage", "Training", "Evaluation", "Limitations"]
    for section in recommended_sections:
        if f"## {section}" not in card_content:
            issues.append(f"Missing '{section}' section")

    # Check for license
    if "license:" not in card_content.lower():
        issues.append("Missing license information")

    is_valid = len(issues) == 0
    return is_valid, issues


def extract_metadata_from_model(model_info: dict[str, Any]) -> ModelMetadata:
    """Extract model metadata from a model info dictionary.

    Args:
        model_info: Dictionary containing model information.

    Returns:
        ModelMetadata extracted from the info.

    Raises:
        ValueError: If required fields are missing.

    Examples:
        >>> info = {
        ...     "modelId": "bert-base",
        ...     "author": "huggingface",
        ...     "tags": ["nlp", "bert"],
        ...     "pipeline_tag": "text_classification",
        ...     "license": "apache2",
        ... }
        >>> metadata = extract_metadata_from_model(info)
        >>> metadata.name
        'bert-base'
        >>> metadata.license
        <LicenseType.APACHE2: 'apache2'>

        >>> extract_metadata_from_model({})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_info must contain 'modelId'
    """
    model_id = model_info.get("modelId") or model_info.get("id")
    if not model_id:
        msg = "model_info must contain 'modelId' or 'id'"
        raise ValueError(msg)

    author = model_info.get("author", "unknown")
    if "/" in model_id and author == "unknown":
        author = model_id.split("/")[0]

    # Parse license
    license_str = model_info.get("license", "mit")
    if license_str not in VALID_LICENSE_TYPES:
        # Try to map common license names
        license_mapping = {
            "apache-2.0": "apache2",
            "cc-by-4.0": "cc_by",
            "cc-by-nc-4.0": "cc_by_nc",
            "openrail-m": "openrail",
            "openrail++": "openrail",
        }
        license_str = license_mapping.get(license_str, "mit")

    # Parse tags
    tags = model_info.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]

    # Parse pipeline_tag
    pipeline_tag_str = model_info.get("pipeline_tag")
    pipeline_tag = None
    if pipeline_tag_str:
        # Map HF pipeline tags to our enum
        pipeline_mapping = {
            "text-classification": "text_classification",
            "text-generation": "text_generation",
            "question-answering": "question_answering",
            "summarization": "summarization",
            "translation": "translation",
        }
        mapped = pipeline_mapping.get(pipeline_tag_str, pipeline_tag_str)
        if mapped in VALID_MODEL_TASKS:
            pipeline_tag = mapped

    return create_model_metadata(
        name=model_id,
        version="1.0.0",
        author=author,
        license_type=license_str,
        tags=tags,
        pipeline_tag=pipeline_tag,
    )


def format_model_card_stats(stats: ModelCardStats) -> str:
    """Format model card statistics as a human-readable string.

    Args:
        stats: Model card statistics.

    Returns:
        Formatted statistics string.

    Examples:
        >>> stats = ModelCardStats(
        ...     completeness_score=0.85,
        ...     missing_sections=("ethics",),
        ...     word_count=500,
        ... )
        >>> output = format_model_card_stats(stats)
        >>> "85.0%" in output
        True
        >>> "500 words" in output
        True

        >>> stats = ModelCardStats(1.0, (), 1000)
        >>> output = format_model_card_stats(stats)
        >>> "100.0%" in output
        True
    """
    lines = [
        "Model Card Statistics:",
        f"  Completeness: {stats.completeness_score * 100:.1f}%",
        f"  Word Count: {stats.word_count} words",
    ]

    if stats.missing_sections:
        lines.append("  Missing Sections:")
        for section in stats.missing_sections:
            lines.append(f"    - {section}")
    else:
        lines.append("  All recommended sections present")

    return "\n".join(lines)


def get_recommended_model_card_config(
    model_name: str,
    task_type: str = "text_classification",
    include_ethics: bool = True,
) -> ModelCardConfig:
    """Get a recommended model card configuration for a model.

    Args:
        model_name: Name of the model.
        task_type: Task type for the model. Defaults to "text_classification".
        include_ethics: Whether to include ethics section. Defaults to True.

    Returns:
        Recommended ModelCardConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_model_card_config("my-bert")
        >>> config.metadata.name
        'my-bert'
        >>> CardSection.ETHICS in config.sections
        True

        >>> config = get_recommended_model_card_config(
        ...     "gpt-clone",
        ...     task_type="text_generation",
        ...     include_ethics=False,
        ... )
        >>> config.metadata.pipeline_tag
        <ModelTask.TEXT_GENERATION: 'text_generation'>
        >>> CardSection.ETHICS in config.sections
        False

        >>> get_recommended_model_card_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     ""
        ... )
        Traceback (most recent call last):
        ValueError: model_name cannot be empty

        >>> get_recommended_model_card_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "x", task_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: task_type must be one of
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if task_type not in VALID_MODEL_TASKS:
        msg = f"task_type must be one of {VALID_MODEL_TASKS}, got '{task_type}'"
        raise ValueError(msg)

    metadata = create_model_metadata(
        name=model_name,
        version="1.0.0",
        author="unknown",
        license_type="apache2",
        tags=[task_type.replace("_", "-")],
        pipeline_tag=task_type,
    )

    sections = [
        CardSection.OVERVIEW,
        CardSection.USAGE,
        CardSection.TRAINING,
        CardSection.EVALUATION,
        CardSection.LIMITATIONS,
    ]

    if include_ethics:
        sections.append(CardSection.ETHICS)

    return create_model_card_config(
        metadata=metadata,
        sections=sections,
        language=("en",),
    )
