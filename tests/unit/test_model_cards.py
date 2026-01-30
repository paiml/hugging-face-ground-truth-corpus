"""Tests for hub.model_cards module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.hub.model_cards import (
    VALID_CARD_SECTIONS,
    VALID_LICENSE_TYPES,
    VALID_MODEL_TASKS,
    CardSection,
    EvaluationResults,
    LicenseType,
    ModelCardConfig,
    ModelCardStats,
    ModelMetadata,
    ModelTask,
    TrainingDetails,
    calculate_completeness,
    create_evaluation_results,
    create_model_card_config,
    create_model_metadata,
    create_training_details,
    extract_metadata_from_model,
    format_model_card_stats,
    generate_model_card,
    get_card_section,
    get_license_type,
    get_model_task,
    get_recommended_model_card_config,
    list_card_sections,
    list_license_types,
    list_model_tasks,
    validate_card_structure,
    validate_evaluation_results,
    validate_model_card_config,
    validate_model_metadata,
    validate_training_details,
)


class TestCardSection:
    """Tests for CardSection enum."""

    def test_all_sections_have_values(self) -> None:
        """All sections have string values."""
        for section in CardSection:
            assert isinstance(section.value, str)

    def test_overview_value(self) -> None:
        """Overview has correct value."""
        assert CardSection.OVERVIEW.value == "overview"

    def test_usage_value(self) -> None:
        """Usage has correct value."""
        assert CardSection.USAGE.value == "usage"

    def test_training_value(self) -> None:
        """Training has correct value."""
        assert CardSection.TRAINING.value == "training"

    def test_evaluation_value(self) -> None:
        """Evaluation has correct value."""
        assert CardSection.EVALUATION.value == "evaluation"

    def test_limitations_value(self) -> None:
        """Limitations has correct value."""
        assert CardSection.LIMITATIONS.value == "limitations"

    def test_ethics_value(self) -> None:
        """Ethics has correct value."""
        assert CardSection.ETHICS.value == "ethics"

    def test_valid_sections_frozenset(self) -> None:
        """VALID_CARD_SECTIONS is a frozenset."""
        assert isinstance(VALID_CARD_SECTIONS, frozenset)

    def test_valid_sections_contains_all(self) -> None:
        """VALID_CARD_SECTIONS contains all enum values."""
        for section in CardSection:
            assert section.value in VALID_CARD_SECTIONS


class TestLicenseType:
    """Tests for LicenseType enum."""

    def test_all_licenses_have_values(self) -> None:
        """All licenses have string values."""
        for license_type in LicenseType:
            assert isinstance(license_type.value, str)

    def test_mit_value(self) -> None:
        """MIT has correct value."""
        assert LicenseType.MIT.value == "mit"

    def test_apache2_value(self) -> None:
        """Apache2 has correct value."""
        assert LicenseType.APACHE2.value == "apache2"

    def test_cc_by_value(self) -> None:
        """CC_BY has correct value."""
        assert LicenseType.CC_BY.value == "cc_by"

    def test_cc_by_nc_value(self) -> None:
        """CC_BY_NC has correct value."""
        assert LicenseType.CC_BY_NC.value == "cc_by_nc"

    def test_proprietary_value(self) -> None:
        """Proprietary has correct value."""
        assert LicenseType.PROPRIETARY.value == "proprietary"

    def test_openrail_value(self) -> None:
        """OpenRAIL has correct value."""
        assert LicenseType.OPENRAIL.value == "openrail"

    def test_valid_licenses_frozenset(self) -> None:
        """VALID_LICENSE_TYPES is a frozenset."""
        assert isinstance(VALID_LICENSE_TYPES, frozenset)


class TestModelTask:
    """Tests for ModelTask enum."""

    def test_all_tasks_have_values(self) -> None:
        """All tasks have string values."""
        for task in ModelTask:
            assert isinstance(task.value, str)

    def test_text_classification_value(self) -> None:
        """Text classification has correct value."""
        assert ModelTask.TEXT_CLASSIFICATION.value == "text_classification"

    def test_text_generation_value(self) -> None:
        """Text generation has correct value."""
        assert ModelTask.TEXT_GENERATION.value == "text_generation"

    def test_question_answering_value(self) -> None:
        """Question answering has correct value."""
        assert ModelTask.QUESTION_ANSWERING.value == "question_answering"

    def test_summarization_value(self) -> None:
        """Summarization has correct value."""
        assert ModelTask.SUMMARIZATION.value == "summarization"

    def test_translation_value(self) -> None:
        """Translation has correct value."""
        assert ModelTask.TRANSLATION.value == "translation"

    def test_valid_tasks_frozenset(self) -> None:
        """VALID_MODEL_TASKS is a frozenset."""
        assert isinstance(VALID_MODEL_TASKS, frozenset)


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Create model metadata."""
        metadata = ModelMetadata(
            name="bert-base",
            version="1.0.0",
            author="huggingface",
            license=LicenseType.APACHE2,
            tags=("nlp", "bert"),
            pipeline_tag=ModelTask.TEXT_CLASSIFICATION,
        )
        assert metadata.name == "bert-base"
        assert metadata.license == LicenseType.APACHE2

    def test_metadata_is_frozen(self) -> None:
        """Metadata is immutable."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        with pytest.raises(AttributeError):
            metadata.name = "new-name"  # type: ignore[misc]

    def test_metadata_has_slots(self) -> None:
        """Metadata uses __slots__."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        assert not hasattr(metadata, "__dict__")


class TestTrainingDetails:
    """Tests for TrainingDetails dataclass."""

    def test_create_details(self) -> None:
        """Create training details."""
        details = TrainingDetails(
            dataset="wikipedia",
            epochs=3,
            batch_size=32,
            learning_rate=2e-5,
            hardware="8x A100 GPUs",
        )
        assert details.dataset == "wikipedia"
        assert details.epochs == 3

    def test_details_is_frozen(self) -> None:
        """Details is immutable."""
        details = TrainingDetails("wiki", 3, 32, 2e-5, "GPU")
        with pytest.raises(AttributeError):
            details.epochs = 10  # type: ignore[misc]

    def test_details_has_slots(self) -> None:
        """Details uses __slots__."""
        details = TrainingDetails("wiki", 3, 32, 2e-5, "GPU")
        assert not hasattr(details, "__dict__")


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""

    def test_create_results(self) -> None:
        """Create evaluation results."""
        results = EvaluationResults(
            metrics=("accuracy", "f1"),
            datasets=("glue", "superglue"),
            scores={"accuracy": 0.92, "f1": 0.89},
        )
        assert results.scores["accuracy"] == 0.92
        assert "f1" in results.metrics

    def test_results_is_frozen(self) -> None:
        """Results is immutable."""
        results = EvaluationResults(
            ("accuracy",), ("glue",), {"accuracy": 0.9}
        )
        with pytest.raises(AttributeError):
            results.metrics = ("f1",)  # type: ignore[misc]

    def test_results_has_slots(self) -> None:
        """Results uses __slots__."""
        results = EvaluationResults(
            ("accuracy",), ("glue",), {"accuracy": 0.9}
        )
        assert not hasattr(results, "__dict__")


class TestModelCardConfig:
    """Tests for ModelCardConfig dataclass."""

    def test_create_config(self) -> None:
        """Create model card config."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        config = ModelCardConfig(
            metadata=metadata,
            training=None,
            evaluation=None,
            sections=(CardSection.OVERVIEW, CardSection.USAGE),
            language=("en",),
        )
        assert config.language == ("en",)
        assert len(config.sections) == 2

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        config = ModelCardConfig(metadata, None, None, (), ("en",))
        with pytest.raises(AttributeError):
            config.language = ("de",)  # type: ignore[misc]


class TestModelCardStats:
    """Tests for ModelCardStats dataclass."""

    def test_create_stats(self) -> None:
        """Create model card stats."""
        stats = ModelCardStats(
            completeness_score=0.85,
            missing_sections=("ethics",),
            word_count=500,
        )
        assert stats.completeness_score == 0.85
        assert "ethics" in stats.missing_sections

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = ModelCardStats(0.85, ("ethics",), 500)
        with pytest.raises(AttributeError):
            stats.word_count = 600  # type: ignore[misc]


class TestValidateModelMetadata:
    """Tests for validate_model_metadata function."""

    def test_valid_metadata(self) -> None:
        """Valid metadata passes validation."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        validate_model_metadata(metadata)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        metadata = ModelMetadata(
            "", "1.0", "hf", LicenseType.MIT, (), None
        )
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_model_metadata(metadata)

    def test_empty_version_raises(self) -> None:
        """Empty version raises ValueError."""
        metadata = ModelMetadata(
            "bert", "", "hf", LicenseType.MIT, (), None
        )
        with pytest.raises(ValueError, match="version cannot be empty"):
            validate_model_metadata(metadata)

    def test_empty_author_raises(self) -> None:
        """Empty author raises ValueError."""
        metadata = ModelMetadata(
            "bert", "1.0", "", LicenseType.MIT, (), None
        )
        with pytest.raises(ValueError, match="author cannot be empty"):
            validate_model_metadata(metadata)


class TestValidateTrainingDetails:
    """Tests for validate_training_details function."""

    def test_valid_details(self) -> None:
        """Valid details passes validation."""
        details = TrainingDetails("wiki", 3, 32, 2e-5, "GPU")
        validate_training_details(details)

    def test_empty_dataset_raises(self) -> None:
        """Empty dataset raises ValueError."""
        details = TrainingDetails("", 3, 32, 2e-5, "GPU")
        with pytest.raises(ValueError, match="dataset cannot be empty"):
            validate_training_details(details)

    def test_zero_epochs_raises(self) -> None:
        """Zero epochs raises ValueError."""
        details = TrainingDetails("wiki", 0, 32, 2e-5, "GPU")
        with pytest.raises(ValueError, match="epochs must be positive"):
            validate_training_details(details)

    def test_negative_epochs_raises(self) -> None:
        """Negative epochs raises ValueError."""
        details = TrainingDetails("wiki", -1, 32, 2e-5, "GPU")
        with pytest.raises(ValueError, match="epochs must be positive"):
            validate_training_details(details)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        details = TrainingDetails("wiki", 3, 0, 2e-5, "GPU")
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_training_details(details)

    def test_zero_learning_rate_raises(self) -> None:
        """Zero learning_rate raises ValueError."""
        details = TrainingDetails("wiki", 3, 32, 0, "GPU")
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_training_details(details)

    def test_empty_hardware_raises(self) -> None:
        """Empty hardware raises ValueError."""
        details = TrainingDetails("wiki", 3, 32, 2e-5, "")
        with pytest.raises(ValueError, match="hardware cannot be empty"):
            validate_training_details(details)


class TestValidateEvaluationResults:
    """Tests for validate_evaluation_results function."""

    def test_valid_results(self) -> None:
        """Valid results passes validation."""
        results = EvaluationResults(
            ("accuracy",), ("glue",), {"accuracy": 0.9}
        )
        validate_evaluation_results(results)

    def test_empty_metrics_raises(self) -> None:
        """Empty metrics raises ValueError."""
        results = EvaluationResults((), ("glue",), {})
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            validate_evaluation_results(results)

    def test_empty_datasets_raises(self) -> None:
        """Empty datasets raises ValueError."""
        results = EvaluationResults(
            ("accuracy",), (), {"accuracy": 0.9}
        )
        with pytest.raises(ValueError, match="datasets cannot be empty"):
            validate_evaluation_results(results)

    def test_missing_score_raises(self) -> None:
        """Missing score for metric raises ValueError."""
        results = EvaluationResults(
            ("accuracy", "f1"), ("glue",), {"accuracy": 0.9}
        )
        with pytest.raises(ValueError, match="metric 'f1' not found in scores"):
            validate_evaluation_results(results)

    def test_non_numeric_score_raises(self) -> None:
        """Non-numeric score raises ValueError."""
        results = EvaluationResults(
            ("accuracy",), ("glue",), {"accuracy": "not a number"}  # type: ignore[dict-item]
        )
        with pytest.raises(ValueError, match="score for 'accuracy' must be numeric"):
            validate_evaluation_results(results)


class TestValidateModelCardConfig:
    """Tests for validate_model_card_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        config = ModelCardConfig(metadata, None, None, (), ("en",))
        validate_model_card_config(config)

    def test_invalid_metadata_raises(self) -> None:
        """Invalid metadata raises ValueError."""
        metadata = ModelMetadata(
            "", "1.0", "hf", LicenseType.MIT, (), None
        )
        config = ModelCardConfig(metadata, None, None, (), ("en",))
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_model_card_config(config)

    def test_empty_language_raises(self) -> None:
        """Empty language raises ValueError."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        config = ModelCardConfig(metadata, None, None, (), ())
        with pytest.raises(ValueError, match="language cannot be empty"):
            validate_model_card_config(config)

    def test_empty_language_entry_raises(self) -> None:
        """Empty language entry raises ValueError."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        config = ModelCardConfig(metadata, None, None, (), ("en", ""))
        with pytest.raises(ValueError, match="language entries cannot be empty"):
            validate_model_card_config(config)

    def test_with_valid_training(self) -> None:
        """Config with valid training passes."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        training = TrainingDetails("wiki", 3, 32, 2e-5, "GPU")
        config = ModelCardConfig(metadata, training, None, (), ("en",))
        validate_model_card_config(config)

    def test_with_invalid_training_raises(self) -> None:
        """Config with invalid training raises."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        training = TrainingDetails("", 3, 32, 2e-5, "GPU")
        config = ModelCardConfig(metadata, training, None, (), ("en",))
        with pytest.raises(ValueError, match="dataset cannot be empty"):
            validate_model_card_config(config)

    def test_with_valid_evaluation(self) -> None:
        """Config with valid evaluation passes."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        evaluation = EvaluationResults(
            ("accuracy",), ("glue",), {"accuracy": 0.9}
        )
        config = ModelCardConfig(metadata, None, evaluation, (), ("en",))
        validate_model_card_config(config)

    def test_with_invalid_evaluation_raises(self) -> None:
        """Config with invalid evaluation raises."""
        metadata = ModelMetadata(
            "bert", "1.0", "hf", LicenseType.MIT, (), None
        )
        evaluation = EvaluationResults((), (), {})
        config = ModelCardConfig(metadata, None, evaluation, (), ("en",))
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            validate_model_card_config(config)


class TestCreateModelMetadata:
    """Tests for create_model_metadata function."""

    def test_default_metadata(self) -> None:
        """Create default metadata."""
        metadata = create_model_metadata("bert-base")
        assert metadata.name == "bert-base"
        assert metadata.version == "1.0.0"
        assert metadata.author == "unknown"
        assert metadata.license == LicenseType.MIT

    def test_custom_metadata(self) -> None:
        """Create custom metadata."""
        metadata = create_model_metadata(
            "gpt2",
            version="2.0.0",
            author="openai",
            license_type="apache2",
            tags=["nlp", "generation"],
            pipeline_tag="text_generation",
        )
        assert metadata.version == "2.0.0"
        assert metadata.author == "openai"
        assert metadata.license == LicenseType.APACHE2
        assert metadata.tags == ("nlp", "generation")
        assert metadata.pipeline_tag == ModelTask.TEXT_GENERATION

    def test_tuple_tags(self) -> None:
        """Tags can be passed as tuple."""
        metadata = create_model_metadata("bert", tags=("a", "b"))
        assert metadata.tags == ("a", "b")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_model_metadata("")

    def test_invalid_license_raises(self) -> None:
        """Invalid license raises ValueError."""
        with pytest.raises(ValueError, match="license_type must be one of"):
            create_model_metadata("bert", license_type="invalid")

    def test_invalid_pipeline_tag_raises(self) -> None:
        """Invalid pipeline_tag raises ValueError."""
        with pytest.raises(ValueError, match="pipeline_tag must be one of"):
            create_model_metadata("bert", pipeline_tag="invalid")


class TestCreateTrainingDetails:
    """Tests for create_training_details function."""

    def test_default_details(self) -> None:
        """Create default training details."""
        details = create_training_details("wikipedia")
        assert details.dataset == "wikipedia"
        assert details.epochs == 3
        assert details.batch_size == 32
        assert details.learning_rate == 2e-5
        assert details.hardware == "GPU"

    def test_custom_details(self) -> None:
        """Create custom training details."""
        details = create_training_details(
            "imdb",
            epochs=5,
            batch_size=64,
            learning_rate=1e-4,
            hardware="8x A100",
        )
        assert details.dataset == "imdb"
        assert details.epochs == 5
        assert details.batch_size == 64
        assert details.learning_rate == 1e-4
        assert details.hardware == "8x A100"

    def test_empty_dataset_raises(self) -> None:
        """Empty dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be empty"):
            create_training_details("")

    def test_zero_epochs_raises(self) -> None:
        """Zero epochs raises ValueError."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            create_training_details("wiki", epochs=0)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_training_details("wiki", batch_size=0)


class TestCreateEvaluationResults:
    """Tests for create_evaluation_results function."""

    def test_create_results(self) -> None:
        """Create evaluation results."""
        results = create_evaluation_results(
            ["accuracy", "f1"],
            ["glue"],
            {"accuracy": 0.92, "f1": 0.89},
        )
        assert results.scores["accuracy"] == 0.92
        assert results.metrics == ("accuracy", "f1")
        assert results.datasets == ("glue",)

    def test_tuple_inputs(self) -> None:
        """Can pass tuples for metrics and datasets."""
        results = create_evaluation_results(
            ("accuracy",),
            ("glue", "superglue"),
            {"accuracy": 0.9},
        )
        assert results.metrics == ("accuracy",)
        assert results.datasets == ("glue", "superglue")

    def test_empty_metrics_raises(self) -> None:
        """Empty metrics raises ValueError."""
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            create_evaluation_results([], ["glue"], {})

    def test_missing_score_raises(self) -> None:
        """Missing score for metric raises ValueError."""
        with pytest.raises(ValueError, match="metric 'acc' not found in scores"):
            create_evaluation_results(["acc"], ["glue"], {})


class TestCreateModelCardConfig:
    """Tests for create_model_card_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(metadata)
        assert config.language == ("en",)
        assert len(config.sections) == len(CardSection)

    def test_custom_config(self) -> None:
        """Create custom config."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(
            metadata,
            sections=[CardSection.OVERVIEW, CardSection.USAGE],
            language=["en", "de"],
        )
        assert config.language == ("en", "de")
        assert len(config.sections) == 2

    def test_with_training(self) -> None:
        """Create config with training details."""
        metadata = create_model_metadata("bert")
        training = create_training_details("wiki")
        config = create_model_card_config(metadata, training=training)
        assert config.training is not None
        assert config.training.dataset == "wiki"

    def test_with_evaluation(self) -> None:
        """Create config with evaluation results."""
        metadata = create_model_metadata("bert")
        evaluation = create_evaluation_results(
            ["accuracy"], ["glue"], {"accuracy": 0.9}
        )
        config = create_model_card_config(metadata, evaluation=evaluation)
        assert config.evaluation is not None
        assert config.evaluation.scores["accuracy"] == 0.9


class TestListCardSections:
    """Tests for list_card_sections function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        sections = list_card_sections()
        assert sections == sorted(sections)

    def test_contains_overview(self) -> None:
        """Contains overview."""
        sections = list_card_sections()
        assert "overview" in sections

    def test_contains_ethics(self) -> None:
        """Contains ethics."""
        sections = list_card_sections()
        assert "ethics" in sections


class TestListLicenseTypes:
    """Tests for list_license_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        licenses = list_license_types()
        assert licenses == sorted(licenses)

    def test_contains_mit(self) -> None:
        """Contains mit."""
        licenses = list_license_types()
        assert "mit" in licenses

    def test_contains_apache2(self) -> None:
        """Contains apache2."""
        licenses = list_license_types()
        assert "apache2" in licenses


class TestListModelTasks:
    """Tests for list_model_tasks function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        tasks = list_model_tasks()
        assert tasks == sorted(tasks)

    def test_contains_text_classification(self) -> None:
        """Contains text_classification."""
        tasks = list_model_tasks()
        assert "text_classification" in tasks

    def test_contains_text_generation(self) -> None:
        """Contains text_generation."""
        tasks = list_model_tasks()
        assert "text_generation" in tasks


class TestGetCardSection:
    """Tests for get_card_section function."""

    def test_get_overview(self) -> None:
        """Get overview section."""
        assert get_card_section("overview") == CardSection.OVERVIEW

    def test_get_training(self) -> None:
        """Get training section."""
        assert get_card_section("training") == CardSection.TRAINING

    def test_get_ethics(self) -> None:
        """Get ethics section."""
        assert get_card_section("ethics") == CardSection.ETHICS

    def test_invalid_section_raises(self) -> None:
        """Invalid section raises ValueError."""
        with pytest.raises(ValueError, match="card_section must be one of"):
            get_card_section("invalid")


class TestGetLicenseType:
    """Tests for get_license_type function."""

    def test_get_mit(self) -> None:
        """Get MIT license."""
        assert get_license_type("mit") == LicenseType.MIT

    def test_get_apache2(self) -> None:
        """Get Apache2 license."""
        assert get_license_type("apache2") == LicenseType.APACHE2

    def test_get_openrail(self) -> None:
        """Get OpenRAIL license."""
        assert get_license_type("openrail") == LicenseType.OPENRAIL

    def test_invalid_license_raises(self) -> None:
        """Invalid license raises ValueError."""
        with pytest.raises(ValueError, match="license_type must be one of"):
            get_license_type("invalid")


class TestGetModelTask:
    """Tests for get_model_task function."""

    def test_get_text_classification(self) -> None:
        """Get text classification task."""
        assert get_model_task("text_classification") == ModelTask.TEXT_CLASSIFICATION

    def test_get_text_generation(self) -> None:
        """Get text generation task."""
        assert get_model_task("text_generation") == ModelTask.TEXT_GENERATION

    def test_get_translation(self) -> None:
        """Get translation task."""
        assert get_model_task("translation") == ModelTask.TRANSLATION

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="model_task must be one of"):
            get_model_task("invalid")


class TestGenerateModelCard:
    """Tests for generate_model_card function."""

    def test_generate_basic_card(self) -> None:
        """Generate basic model card."""
        metadata = create_model_metadata("bert-base", author="huggingface")
        config = create_model_card_config(metadata)
        card = generate_model_card(config)
        assert "bert-base" in card
        assert "---" in card
        assert "license: mit" in card

    def test_generate_card_with_training(self) -> None:
        """Generate card with training details."""
        metadata = create_model_metadata("bert")
        training = create_training_details("wikipedia", epochs=5)
        config = create_model_card_config(metadata, training=training)
        card = generate_model_card(config)
        assert "wikipedia" in card
        assert "5" in card

    def test_generate_card_with_evaluation(self) -> None:
        """Generate card with evaluation results."""
        metadata = create_model_metadata("bert")
        evaluation = create_evaluation_results(
            ["accuracy"], ["glue"], {"accuracy": 0.92}
        )
        config = create_model_card_config(metadata, evaluation=evaluation)
        card = generate_model_card(config)
        assert "0.92" in card or "0.9200" in card
        assert "glue" in card

    def test_generate_card_with_tags(self) -> None:
        """Generate card with tags."""
        metadata = create_model_metadata("bert", tags=["nlp", "bert"])
        config = create_model_card_config(metadata)
        card = generate_model_card(config)
        assert "nlp" in card
        assert "bert" in card

    def test_generate_card_with_pipeline_tag(self) -> None:
        """Generate card with pipeline tag."""
        metadata = create_model_metadata(
            "bert", pipeline_tag="text_classification"
        )
        config = create_model_card_config(metadata)
        card = generate_model_card(config)
        assert "text_classification" in card or "text-classification" in card

    def test_generate_card_with_multiple_languages(self) -> None:
        """Generate card with multiple languages."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(metadata, language=["en", "de"])
        card = generate_model_card(config)
        assert "en" in card
        assert "de" in card

    def test_generate_card_has_frontmatter(self) -> None:
        """Generated card has YAML frontmatter."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(metadata)
        card = generate_model_card(config)
        assert card.startswith("---")

    def test_generate_card_has_sections(self) -> None:
        """Generated card has sections."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(metadata)
        card = generate_model_card(config)
        assert "## Usage" in card
        assert "## Training" in card
        assert "## Evaluation" in card
        assert "## Limitations" in card

    def test_generate_card_subset_sections(self) -> None:
        """Generate card with subset of sections."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(
            metadata, sections=[CardSection.OVERVIEW, CardSection.USAGE]
        )
        card = generate_model_card(config)
        assert "## Usage" in card
        assert "## Training" not in card


class TestCalculateCompleteness:
    """Tests for calculate_completeness function."""

    def test_basic_completeness(self) -> None:
        """Calculate basic completeness."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(metadata)
        score = calculate_completeness(config)
        assert 0.0 <= score <= 1.0

    def test_higher_with_training(self) -> None:
        """Score is higher with training details."""
        metadata = create_model_metadata("bert")
        config_without = create_model_card_config(metadata)
        training = create_training_details("wiki")
        config_with = create_model_card_config(metadata, training=training)
        assert calculate_completeness(config_with) > calculate_completeness(
            config_without
        )

    def test_higher_with_evaluation(self) -> None:
        """Score is higher with evaluation results."""
        metadata = create_model_metadata("bert")
        config_without = create_model_card_config(metadata)
        evaluation = create_evaluation_results(
            ["acc"], ["glue"], {"acc": 0.9}
        )
        config_with = create_model_card_config(metadata, evaluation=evaluation)
        assert calculate_completeness(config_with) > calculate_completeness(
            config_without
        )

    def test_higher_with_tags(self) -> None:
        """Score is higher with tags."""
        metadata_without = create_model_metadata("bert")
        metadata_with = create_model_metadata("bert", tags=["nlp"])
        config_without = create_model_card_config(metadata_without)
        config_with = create_model_card_config(metadata_with)
        assert calculate_completeness(config_with) > calculate_completeness(
            config_without
        )

    def test_higher_with_pipeline_tag(self) -> None:
        """Score is higher with pipeline tag."""
        metadata_without = create_model_metadata("bert")
        metadata_with = create_model_metadata(
            "bert", pipeline_tag="text_classification"
        )
        config_without = create_model_card_config(metadata_without)
        config_with = create_model_card_config(metadata_with)
        assert calculate_completeness(config_with) > calculate_completeness(
            config_without
        )


class TestValidateCardStructure:
    """Tests for validate_card_structure function."""

    def test_valid_generated_card(self) -> None:
        """Generated card passes validation."""
        metadata = create_model_metadata("bert")
        config = create_model_card_config(metadata)
        card = generate_model_card(config)
        is_valid, issues = validate_card_structure(card)
        assert is_valid
        assert len(issues) == 0

    def test_minimal_card_fails(self) -> None:
        """Minimal card fails validation."""
        is_valid, issues = validate_card_structure("# Minimal Card")
        assert not is_valid
        assert len(issues) > 0

    def test_missing_frontmatter(self) -> None:
        """Missing frontmatter is detected."""
        is_valid, issues = validate_card_structure("# Card\n## Usage\nText.")
        assert not is_valid
        assert any("frontmatter" in i.lower() for i in issues)

    def test_empty_content_raises(self) -> None:
        """Empty content raises ValueError."""
        with pytest.raises(ValueError, match="card_content cannot be empty"):
            validate_card_structure("")

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only content raises ValueError."""
        with pytest.raises(ValueError, match="card_content cannot be empty"):
            validate_card_structure("   \n\n   ")

    def test_missing_license(self) -> None:
        """Missing license is detected."""
        card = "---\ntags:\n  - nlp\n---\n# Card"
        is_valid, issues = validate_card_structure(card)
        assert not is_valid
        assert any("license" in i.lower() for i in issues)

    def test_missing_title_detected(self) -> None:
        """Missing title is detected."""
        card = "---\nlicense: mit\n---\nNo heading here"
        is_valid, issues = validate_card_structure(card)
        assert not is_valid
        assert any("title" in i.lower() for i in issues)

    def test_card_starting_with_heading(self) -> None:
        """Card starting with # heading is valid for title check."""
        card = "# Title\n## Usage\n## Training\n## Evaluation\n## Limitations"
        _is_valid, issues = validate_card_structure(card)
        # Should have frontmatter and license issues but not title
        assert not any("title" in i.lower() for i in issues)


class TestExtractMetadataFromModel:
    """Tests for extract_metadata_from_model function."""

    def test_extract_basic_info(self) -> None:
        """Extract basic model info."""
        info = {
            "modelId": "bert-base",
            "author": "huggingface",
            "tags": ["nlp", "bert"],
            "pipeline_tag": "text-classification",
            "license": "apache2",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.name == "bert-base"
        assert metadata.author == "huggingface"
        assert metadata.license == LicenseType.APACHE2
        assert metadata.pipeline_tag == ModelTask.TEXT_CLASSIFICATION

    def test_extract_with_id_key(self) -> None:
        """Extract with 'id' key instead of 'modelId'."""
        info = {
            "id": "gpt2",
            "author": "openai",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.name == "gpt2"
        assert metadata.author == "openai"

    def test_extract_author_from_model_id(self) -> None:
        """Extract author from model ID if not provided."""
        info = {
            "modelId": "huggingface/bert-base",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.author == "huggingface"

    def test_missing_model_id_raises(self) -> None:
        """Missing model ID raises ValueError."""
        with pytest.raises(ValueError, match="model_info must contain"):
            extract_metadata_from_model({})

    def test_license_mapping(self) -> None:
        """Licenses are mapped correctly."""
        info = {
            "modelId": "bert",
            "license": "apache-2.0",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.license == LicenseType.APACHE2

    def test_cc_license_mapping(self) -> None:
        """CC licenses are mapped correctly."""
        info = {
            "modelId": "bert",
            "license": "cc-by-4.0",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.license == LicenseType.CC_BY

    def test_openrail_license_mapping(self) -> None:
        """OpenRAIL licenses are mapped correctly."""
        info = {
            "modelId": "bert",
            "license": "openrail-m",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.license == LicenseType.OPENRAIL

    def test_unknown_license_defaults_to_mit(self) -> None:
        """Unknown license defaults to MIT."""
        info = {
            "modelId": "bert",
            "license": "unknown-license",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.license == LicenseType.MIT

    def test_string_tags_converted_to_list(self) -> None:
        """String tags are converted to list."""
        info = {
            "modelId": "bert",
            "tags": "nlp",
        }
        metadata = extract_metadata_from_model(info)
        assert metadata.tags == ("nlp",)


class TestFormatModelCardStats:
    """Tests for format_model_card_stats function."""

    def test_format_basic_stats(self) -> None:
        """Format basic stats."""
        stats = ModelCardStats(
            completeness_score=0.85,
            missing_sections=("ethics",),
            word_count=500,
        )
        output = format_model_card_stats(stats)
        assert "85.0%" in output
        assert "500 words" in output
        assert "ethics" in output

    def test_format_complete_stats(self) -> None:
        """Format complete stats."""
        stats = ModelCardStats(1.0, (), 1000)
        output = format_model_card_stats(stats)
        assert "100.0%" in output
        assert "1000 words" in output
        assert "All recommended sections present" in output

    def test_format_multiple_missing(self) -> None:
        """Format stats with multiple missing sections."""
        stats = ModelCardStats(
            0.5,
            ("ethics", "limitations", "training"),
            200,
        )
        output = format_model_card_stats(stats)
        assert "ethics" in output
        assert "limitations" in output
        assert "training" in output


class TestGetRecommendedModelCardConfig:
    """Tests for get_recommended_model_card_config function."""

    def test_default_config(self) -> None:
        """Get default recommended config."""
        config = get_recommended_model_card_config("my-bert")
        assert config.metadata.name == "my-bert"
        assert CardSection.ETHICS in config.sections

    def test_text_generation_task(self) -> None:
        """Get config for text generation."""
        config = get_recommended_model_card_config(
            "gpt-clone", task_type="text_generation"
        )
        assert config.metadata.pipeline_tag == ModelTask.TEXT_GENERATION

    def test_without_ethics(self) -> None:
        """Get config without ethics section."""
        config = get_recommended_model_card_config(
            "bert", include_ethics=False
        )
        assert CardSection.ETHICS not in config.sections

    def test_empty_model_name_raises(self) -> None:
        """Empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            get_recommended_model_card_config("")

    def test_invalid_task_type_raises(self) -> None:
        """Invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_model_card_config("bert", task_type="invalid")


class TestPropertyBased:
    """Property-based tests."""

    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters=("-", "_"),
    )))
    @settings(max_examples=20)
    def test_model_name_preserved(self, name: str) -> None:
        """Model name is preserved in metadata."""
        if not name.strip():
            return
        metadata = create_model_metadata(name.strip())
        assert metadata.name == name.strip()

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_epochs_preserved(self, epochs: int) -> None:
        """Epochs are preserved in training details."""
        details = create_training_details("dataset", epochs=epochs)
        assert details.epochs == epochs

    @given(st.floats(min_value=1e-10, max_value=1.0, allow_nan=False))
    @settings(max_examples=20)
    def test_learning_rate_preserved(self, lr: float) -> None:
        """Learning rate is preserved in training details."""
        details = create_training_details("dataset", learning_rate=lr)
        assert details.learning_rate == lr

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=20)
    def test_evaluation_score_preserved(self, score: float) -> None:
        """Evaluation score is preserved."""
        results = create_evaluation_results(
            ["accuracy"], ["dataset"], {"accuracy": score}
        )
        assert results.scores["accuracy"] == score

    @given(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=("L", "N"),
        )),
        min_size=1,
        max_size=5,
        unique=True,
    ))
    @settings(max_examples=10)
    def test_tags_preserved(self, tags: list[str]) -> None:
        """Tags are preserved in metadata."""
        clean_tags = [t.strip() for t in tags if t.strip()]
        if not clean_tags:
            return
        metadata = create_model_metadata("bert", tags=clean_tags)
        assert metadata.tags == tuple(clean_tags)
