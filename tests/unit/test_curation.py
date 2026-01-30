"""Tests for dataset curation and cleaning functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.curation import (
    VALID_CLEANING_METHODS,
    VALID_CURATION_STEPS,
    VALID_DECONTAMINATION_TYPES,
    CleaningConfig,
    CleaningMethod,
    CurationPipeline,
    CurationStats,
    CurationStep,
    DecontaminationConfig,
    DecontaminationType,
    apply_cleaning,
    create_cleaning_config,
    create_curation_pipeline,
    create_decontamination_config,
    detect_contamination,
    estimate_curation_time,
    format_curation_stats,
    get_cleaning_method,
    get_curation_step,
    get_decontamination_type,
    get_recommended_curation_config,
    list_cleaning_methods,
    list_curation_steps,
    list_decontamination_types,
    run_curation_pipeline,
    validate_cleaning_config,
    validate_curation_pipeline,
    validate_curation_stats,
    validate_decontamination_config,
)


class TestCurationStep:
    """Tests for CurationStep enum."""

    def test_dedup_value(self) -> None:
        """Test DEDUP value."""
        assert CurationStep.DEDUP.value == "dedup"

    def test_filter_value(self) -> None:
        """Test FILTER value."""
        assert CurationStep.FILTER.value == "filter"

    def test_clean_value(self) -> None:
        """Test CLEAN value."""
        assert CurationStep.CLEAN.value == "clean"

    def test_normalize_value(self) -> None:
        """Test NORMALIZE value."""
        assert CurationStep.NORMALIZE.value == "normalize"

    def test_validate_value(self) -> None:
        """Test VALIDATE value."""
        assert CurationStep.VALIDATE.value == "validate"


class TestCleaningMethod:
    """Tests for CleaningMethod enum."""

    def test_whitespace_value(self) -> None:
        """Test WHITESPACE value."""
        assert CleaningMethod.WHITESPACE.value == "whitespace"

    def test_unicode_value(self) -> None:
        """Test UNICODE value."""
        assert CleaningMethod.UNICODE.value == "unicode"

    def test_html_value(self) -> None:
        """Test HTML value."""
        assert CleaningMethod.HTML.value == "html"

    def test_markdown_value(self) -> None:
        """Test MARKDOWN value."""
        assert CleaningMethod.MARKDOWN.value == "markdown"


class TestDecontaminationType:
    """Tests for DecontaminationType enum."""

    def test_exact_match_value(self) -> None:
        """Test EXACT_MATCH value."""
        assert DecontaminationType.EXACT_MATCH.value == "exact_match"

    def test_ngram_overlap_value(self) -> None:
        """Test NGRAM_OVERLAP value."""
        assert DecontaminationType.NGRAM_OVERLAP.value == "ngram_overlap"

    def test_embedding_similarity_value(self) -> None:
        """Test EMBEDDING_SIMILARITY value."""
        assert DecontaminationType.EMBEDDING_SIMILARITY.value == "embedding_similarity"


class TestCleaningConfig:
    """Tests for CleaningConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating CleaningConfig instance."""
        config = CleaningConfig(
            methods=(CleaningMethod.WHITESPACE, CleaningMethod.UNICODE),
            lowercase=True,
            strip_accents=False,
            min_length=10,
        )
        assert len(config.methods) == 2
        assert config.lowercase is True
        assert config.strip_accents is False
        assert config.min_length == 10

    def test_frozen(self) -> None:
        """Test that CleaningConfig is immutable."""
        config = CleaningConfig(
            methods=(CleaningMethod.WHITESPACE,),
            lowercase=False,
            strip_accents=False,
            min_length=5,
        )
        with pytest.raises(AttributeError):
            config.lowercase = True  # type: ignore[misc]


class TestDecontaminationConfig:
    """Tests for DecontaminationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating DecontaminationConfig instance."""
        config = DecontaminationConfig(
            decontam_type=DecontaminationType.NGRAM_OVERLAP,
            test_datasets=("squad", "trivia_qa"),
            threshold=0.8,
            ngram_size=5,
        )
        assert config.decontam_type == DecontaminationType.NGRAM_OVERLAP
        assert len(config.test_datasets) == 2
        assert config.threshold == pytest.approx(0.8)
        assert config.ngram_size == 5

    def test_frozen(self) -> None:
        """Test that DecontaminationConfig is immutable."""
        config = DecontaminationConfig(
            decontam_type=DecontaminationType.EXACT_MATCH,
            test_datasets=("squad",),
            threshold=0.8,
            ngram_size=5,
        )
        with pytest.raises(AttributeError):
            config.threshold = 0.9  # type: ignore[misc]


class TestCurationPipeline:
    """Tests for CurationPipeline dataclass."""

    def test_creation(self) -> None:
        """Test creating CurationPipeline instance."""
        cleaning = CleaningConfig(
            methods=(CleaningMethod.WHITESPACE,),
            lowercase=False,
            strip_accents=False,
            min_length=5,
        )
        pipeline = CurationPipeline(
            steps=(CurationStep.CLEAN, CurationStep.DEDUP),
            cleaning_config=cleaning,
            decontam_config=None,
            validate_output=True,
        )
        assert len(pipeline.steps) == 2
        assert pipeline.cleaning_config is not None
        assert pipeline.decontam_config is None
        assert pipeline.validate_output is True

    def test_frozen(self) -> None:
        """Test that CurationPipeline is immutable."""
        pipeline = CurationPipeline(
            steps=(CurationStep.DEDUP,),
            cleaning_config=None,
            decontam_config=None,
            validate_output=True,
        )
        with pytest.raises(AttributeError):
            pipeline.validate_output = False  # type: ignore[misc]


class TestCurationStats:
    """Tests for CurationStats dataclass."""

    def test_creation(self) -> None:
        """Test creating CurationStats instance."""
        stats = CurationStats(
            original_size=10000,
            final_size=8500,
            removed_duplicates=1000,
            removed_contamination=500,
        )
        assert stats.original_size == 10000
        assert stats.final_size == 8500
        assert stats.removed_duplicates == 1000
        assert stats.removed_contamination == 500

    def test_frozen(self) -> None:
        """Test that CurationStats is immutable."""
        stats = CurationStats(
            original_size=1000,
            final_size=800,
            removed_duplicates=100,
            removed_contamination=100,
        )
        with pytest.raises(AttributeError):
            stats.final_size = 700  # type: ignore[misc]


class TestValidateCleaningConfig:
    """Tests for validate_cleaning_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = CleaningConfig(
            methods=(CleaningMethod.WHITESPACE,),
            lowercase=False,
            strip_accents=False,
            min_length=5,
        )
        validate_cleaning_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_cleaning_config(None)  # type: ignore[arg-type]

    def test_empty_methods_raises_error(self) -> None:
        """Test that empty methods raises ValueError."""
        config = CleaningConfig(
            methods=(),
            lowercase=False,
            strip_accents=False,
            min_length=5,
        )
        with pytest.raises(ValueError, match="methods cannot be empty"):
            validate_cleaning_config(config)

    def test_negative_min_length_raises_error(self) -> None:
        """Test that negative min_length raises ValueError."""
        config = CleaningConfig(
            methods=(CleaningMethod.WHITESPACE,),
            lowercase=False,
            strip_accents=False,
            min_length=-1,
        )
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            validate_cleaning_config(config)


class TestValidateDecontaminationConfig:
    """Tests for validate_decontamination_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = DecontaminationConfig(
            decontam_type=DecontaminationType.EXACT_MATCH,
            test_datasets=("squad",),
            threshold=0.8,
            ngram_size=5,
        )
        validate_decontamination_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_decontamination_config(None)  # type: ignore[arg-type]

    def test_empty_test_datasets_raises_error(self) -> None:
        """Test that empty test_datasets raises ValueError."""
        config = DecontaminationConfig(
            decontam_type=DecontaminationType.EXACT_MATCH,
            test_datasets=(),
            threshold=0.8,
            ngram_size=5,
        )
        with pytest.raises(ValueError, match="test_datasets cannot be empty"):
            validate_decontamination_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1 raises ValueError."""
        config = DecontaminationConfig(
            decontam_type=DecontaminationType.EXACT_MATCH,
            test_datasets=("squad",),
            threshold=1.5,
            ngram_size=5,
        )
        with pytest.raises(ValueError, match="threshold must be between"):
            validate_decontamination_config(config)

    def test_threshold_below_zero_raises_error(self) -> None:
        """Test that threshold below 0 raises ValueError."""
        config = DecontaminationConfig(
            decontam_type=DecontaminationType.EXACT_MATCH,
            test_datasets=("squad",),
            threshold=-0.1,
            ngram_size=5,
        )
        with pytest.raises(ValueError, match="threshold must be between"):
            validate_decontamination_config(config)

    def test_non_positive_ngram_size_raises_error(self) -> None:
        """Test that non-positive ngram_size raises ValueError."""
        config = DecontaminationConfig(
            decontam_type=DecontaminationType.EXACT_MATCH,
            test_datasets=("squad",),
            threshold=0.8,
            ngram_size=0,
        )
        with pytest.raises(ValueError, match="ngram_size must be positive"):
            validate_decontamination_config(config)


class TestValidateCurationPipeline:
    """Tests for validate_curation_pipeline function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        cleaning = CleaningConfig(
            methods=(CleaningMethod.WHITESPACE,),
            lowercase=False,
            strip_accents=False,
            min_length=5,
        )
        pipeline = CurationPipeline(
            steps=(CurationStep.CLEAN,),
            cleaning_config=cleaning,
            decontam_config=None,
            validate_output=True,
        )
        validate_curation_pipeline(pipeline)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_curation_pipeline(None)  # type: ignore[arg-type]

    def test_empty_steps_raises_error(self) -> None:
        """Test that empty steps raises ValueError."""
        pipeline = CurationPipeline(
            steps=(),
            cleaning_config=None,
            decontam_config=None,
            validate_output=True,
        )
        with pytest.raises(ValueError, match="steps cannot be empty"):
            validate_curation_pipeline(pipeline)

    def test_clean_step_without_config_raises_error(self) -> None:
        """Test that CLEAN step without cleaning_config raises ValueError."""
        pipeline = CurationPipeline(
            steps=(CurationStep.CLEAN,),
            cleaning_config=None,
            decontam_config=None,
            validate_output=True,
        )
        with pytest.raises(ValueError, match="cleaning_config required"):
            validate_curation_pipeline(pipeline)


class TestValidateCurationStats:
    """Tests for validate_curation_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = CurationStats(
            original_size=1000,
            final_size=800,
            removed_duplicates=100,
            removed_contamination=100,
        )
        validate_curation_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_curation_stats(None)  # type: ignore[arg-type]

    def test_negative_original_size_raises_error(self) -> None:
        """Test that negative original_size raises ValueError."""
        stats = CurationStats(
            original_size=-1,
            final_size=0,
            removed_duplicates=0,
            removed_contamination=0,
        )
        with pytest.raises(ValueError, match="original_size must be non-negative"):
            validate_curation_stats(stats)

    def test_final_greater_than_original_raises_error(self) -> None:
        """Test that final_size > original_size raises ValueError."""
        stats = CurationStats(
            original_size=100,
            final_size=200,
            removed_duplicates=0,
            removed_contamination=0,
        )
        with pytest.raises(ValueError, match="final_size cannot be greater"):
            validate_curation_stats(stats)


class TestCreateCleaningConfig:
    """Tests for create_cleaning_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_cleaning_config()
        assert len(config.methods) == 2
        assert CleaningMethod.WHITESPACE in config.methods
        assert CleaningMethod.UNICODE in config.methods
        assert config.lowercase is False
        assert config.strip_accents is False
        assert config.min_length == 0

    def test_custom_methods(self) -> None:
        """Test creating config with custom methods."""
        config = create_cleaning_config(methods=("html", "markdown"))
        assert len(config.methods) == 2
        assert CleaningMethod.HTML in config.methods
        assert CleaningMethod.MARKDOWN in config.methods

    def test_custom_lowercase(self) -> None:
        """Test creating config with lowercase=True."""
        config = create_cleaning_config(lowercase=True)
        assert config.lowercase is True

    def test_custom_strip_accents(self) -> None:
        """Test creating config with strip_accents=True."""
        config = create_cleaning_config(strip_accents=True)
        assert config.strip_accents is True

    def test_custom_min_length(self) -> None:
        """Test creating config with custom min_length."""
        config = create_cleaning_config(min_length=20)
        assert config.min_length == 20

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_cleaning_config(methods=("invalid",))

    def test_negative_min_length_raises_error(self) -> None:
        """Test that negative min_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            create_cleaning_config(min_length=-1)


class TestCreateDecontaminationConfig:
    """Tests for create_decontamination_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_decontamination_config()
        assert config.decontam_type == DecontaminationType.NGRAM_OVERLAP
        assert "default" in config.test_datasets
        assert config.threshold == pytest.approx(0.8)
        assert config.ngram_size == 5

    def test_custom_decontam_type(self) -> None:
        """Test creating config with custom decontam_type."""
        config = create_decontamination_config(decontam_type="exact_match")
        assert config.decontam_type == DecontaminationType.EXACT_MATCH

    def test_custom_test_datasets(self) -> None:
        """Test creating config with custom test_datasets."""
        config = create_decontamination_config(test_datasets=("squad", "nq"))
        assert len(config.test_datasets) == 2
        assert "squad" in config.test_datasets

    def test_custom_threshold(self) -> None:
        """Test creating config with custom threshold."""
        config = create_decontamination_config(threshold=0.9)
        assert config.threshold == pytest.approx(0.9)

    def test_custom_ngram_size(self) -> None:
        """Test creating config with custom ngram_size."""
        config = create_decontamination_config(ngram_size=7)
        assert config.ngram_size == 7

    def test_invalid_decontam_type_raises_error(self) -> None:
        """Test that invalid decontam_type raises ValueError."""
        with pytest.raises(ValueError, match="decontam_type must be one of"):
            create_decontamination_config(decontam_type="invalid")

    def test_empty_test_datasets_raises_error(self) -> None:
        """Test that empty test_datasets raises ValueError."""
        with pytest.raises(ValueError, match="test_datasets cannot be empty"):
            create_decontamination_config(test_datasets=())


class TestCreateCurationPipeline:
    """Tests for create_curation_pipeline function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        pipeline = create_curation_pipeline()
        assert len(pipeline.steps) == 3
        assert CurationStep.DEDUP in pipeline.steps
        assert CurationStep.CLEAN in pipeline.steps
        assert CurationStep.FILTER in pipeline.steps
        assert pipeline.cleaning_config is not None
        assert pipeline.validate_output is True

    def test_custom_steps(self) -> None:
        """Test creating pipeline with custom steps."""
        pipeline = create_curation_pipeline(steps=("dedup", "validate"))
        assert len(pipeline.steps) == 2
        assert CurationStep.DEDUP in pipeline.steps
        assert CurationStep.VALIDATE in pipeline.steps

    def test_auto_creates_cleaning_config(self) -> None:
        """Test that cleaning_config is auto-created for CLEAN step."""
        pipeline = create_curation_pipeline(steps=("clean",))
        assert pipeline.cleaning_config is not None

    def test_with_decontam_config(self) -> None:
        """Test creating pipeline with decontam_config."""
        decontam = create_decontamination_config()
        pipeline = create_curation_pipeline(
            steps=("dedup",),
            decontam_config=decontam,
        )
        assert pipeline.decontam_config is not None

    def test_custom_validate_output(self) -> None:
        """Test creating pipeline with validate_output=False."""
        pipeline = create_curation_pipeline(
            steps=("dedup",),
            validate_output=False,
        )
        assert pipeline.validate_output is False

    def test_invalid_step_raises_error(self) -> None:
        """Test that invalid step raises ValueError."""
        with pytest.raises(ValueError, match="step must be one of"):
            create_curation_pipeline(steps=("invalid",))


class TestListCurationSteps:
    """Tests for list_curation_steps function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        steps = list_curation_steps()
        assert isinstance(steps, list)

    def test_contains_expected_steps(self) -> None:
        """Test that list contains expected steps."""
        steps = list_curation_steps()
        assert "dedup" in steps
        assert "clean" in steps
        assert "filter" in steps
        assert "normalize" in steps
        assert "validate" in steps

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        steps = list_curation_steps()
        assert steps == sorted(steps)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_CURATION_STEPS."""
        steps = list_curation_steps()
        assert set(steps) == VALID_CURATION_STEPS


class TestGetCurationStep:
    """Tests for get_curation_step function."""

    def test_get_dedup(self) -> None:
        """Test getting DEDUP step."""
        result = get_curation_step("dedup")
        assert result == CurationStep.DEDUP

    def test_get_clean(self) -> None:
        """Test getting CLEAN step."""
        result = get_curation_step("clean")
        assert result == CurationStep.CLEAN

    def test_invalid_step_raises_error(self) -> None:
        """Test that invalid step raises ValueError."""
        with pytest.raises(ValueError, match="invalid curation step"):
            get_curation_step("invalid")


class TestListCleaningMethods:
    """Tests for list_cleaning_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_cleaning_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_cleaning_methods()
        assert "whitespace" in methods
        assert "unicode" in methods
        assert "html" in methods
        assert "markdown" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_cleaning_methods()
        assert methods == sorted(methods)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_CLEANING_METHODS."""
        methods = list_cleaning_methods()
        assert set(methods) == VALID_CLEANING_METHODS


class TestGetCleaningMethod:
    """Tests for get_cleaning_method function."""

    def test_get_whitespace(self) -> None:
        """Test getting WHITESPACE method."""
        result = get_cleaning_method("whitespace")
        assert result == CleaningMethod.WHITESPACE

    def test_get_html(self) -> None:
        """Test getting HTML method."""
        result = get_cleaning_method("html")
        assert result == CleaningMethod.HTML

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid cleaning method"):
            get_cleaning_method("invalid")


class TestListDecontaminationTypes:
    """Tests for list_decontamination_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_decontamination_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_decontamination_types()
        assert "exact_match" in types
        assert "ngram_overlap" in types
        assert "embedding_similarity" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_decontamination_types()
        assert types == sorted(types)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_DECONTAMINATION_TYPES."""
        types = list_decontamination_types()
        assert set(types) == VALID_DECONTAMINATION_TYPES


class TestGetDecontaminationType:
    """Tests for get_decontamination_type function."""

    def test_get_exact_match(self) -> None:
        """Test getting EXACT_MATCH type."""
        result = get_decontamination_type("exact_match")
        assert result == DecontaminationType.EXACT_MATCH

    def test_get_ngram_overlap(self) -> None:
        """Test getting NGRAM_OVERLAP type."""
        result = get_decontamination_type("ngram_overlap")
        assert result == DecontaminationType.NGRAM_OVERLAP

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid decontamination type"):
            get_decontamination_type("invalid")


class TestApplyCleaning:
    """Tests for apply_cleaning function."""

    def test_whitespace_cleaning(self) -> None:
        """Test whitespace cleaning."""
        result = apply_cleaning("  Hello   World  ")
        assert result == "Hello World"

    def test_html_cleaning(self) -> None:
        """Test HTML cleaning."""
        config = create_cleaning_config(methods=("html",))
        result = apply_cleaning("<p>Hello</p>", config)
        assert result == "Hello"

    def test_html_entity_decoding(self) -> None:
        """Test HTML entity decoding."""
        config = create_cleaning_config(methods=("html",))
        result = apply_cleaning("Hello &amp; World", config)
        assert result == "Hello & World"

    def test_markdown_cleaning(self) -> None:
        """Test markdown cleaning."""
        config = create_cleaning_config(methods=("markdown",))
        result = apply_cleaning("**Bold** text", config)
        assert result == "Bold text"

    def test_markdown_italic_cleaning(self) -> None:
        """Test markdown italic cleaning."""
        config = create_cleaning_config(methods=("markdown",))
        result = apply_cleaning("*italic* text", config)
        assert result == "italic text"

    def test_markdown_code_cleaning(self) -> None:
        """Test markdown code cleaning."""
        config = create_cleaning_config(methods=("markdown",))
        result = apply_cleaning("`code` text", config)
        assert result == "code text"

    def test_markdown_link_cleaning(self) -> None:
        """Test markdown link cleaning."""
        config = create_cleaning_config(methods=("markdown",))
        result = apply_cleaning("[link](http://example.com)", config)
        assert result == "link"

    def test_unicode_normalization(self) -> None:
        """Test unicode normalization."""
        config = create_cleaning_config(methods=("unicode",))
        # Combining character form
        result = apply_cleaning("cafe\u0301", config)
        assert len(result) <= 5  # Should be NFC normalized

    def test_lowercase_option(self) -> None:
        """Test lowercase conversion."""
        config = create_cleaning_config(lowercase=True)
        result = apply_cleaning("Hello WORLD", config)
        assert result == "hello world"

    def test_strip_accents_option(self) -> None:
        """Test accent stripping."""
        config = create_cleaning_config(strip_accents=True)
        result = apply_cleaning("cafe", config)
        assert "e" in result

    def test_empty_text(self) -> None:
        """Test with empty text."""
        result = apply_cleaning("")
        assert result == ""

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            apply_cleaning(None)  # type: ignore[arg-type]

    def test_combined_cleaning(self) -> None:
        """Test combined cleaning methods."""
        config = create_cleaning_config(methods=("whitespace", "html"))
        result = apply_cleaning("  <b>Hello</b>   World  ", config)
        assert result == "Hello World"


class TestDetectContamination:
    """Tests for detect_contamination function."""

    def test_exact_match_detection(self) -> None:
        """Test exact match contamination detection."""
        train = ["The quick brown fox", "Hello world"]
        test = ["The quick brown fox", "Different text"]
        config = create_decontamination_config(decontam_type="exact_match")
        contaminated = detect_contamination(train, test, config)
        assert len(contaminated) >= 1
        assert contaminated[0][2] == 1.0  # Exact match has similarity 1.0

    def test_ngram_overlap_detection(self) -> None:
        """Test n-gram overlap contamination detection."""
        train = ["The quick brown fox jumps over the lazy dog"]
        test = ["The quick brown fox jumps over"]
        config = create_decontamination_config(
            decontam_type="ngram_overlap",
            threshold=0.3,
            ngram_size=3,
        )
        contaminated = detect_contamination(train, test, config)
        assert len(contaminated) >= 1

    def test_embedding_similarity_detection(self) -> None:
        """Test embedding similarity detection (falls back to ngram)."""
        train = ["The quick brown fox jumps over the lazy dog today"]
        test = ["The quick brown fox jumps over the lazy dog today"]
        config = create_decontamination_config(
            decontam_type="embedding_similarity",
            threshold=0.5,
            ngram_size=3,
        )
        contaminated = detect_contamination(train, test, config)
        assert len(contaminated) >= 1

    def test_empty_lists(self) -> None:
        """Test with empty lists."""
        result = detect_contamination([], [])
        assert result == []

    def test_none_train_raises_error(self) -> None:
        """Test that None train_texts raises ValueError."""
        with pytest.raises(ValueError, match="train_texts cannot be None"):
            detect_contamination(None, [])  # type: ignore[arg-type]

    def test_none_test_raises_error(self) -> None:
        """Test that None test_texts raises ValueError."""
        with pytest.raises(ValueError, match="test_texts cannot be None"):
            detect_contamination([], None)  # type: ignore[arg-type]

    def test_no_contamination(self) -> None:
        """Test when there is no contamination."""
        train = ["Hello world"]
        test = ["Goodbye universe"]
        config = create_decontamination_config(threshold=0.9)
        contaminated = detect_contamination(train, test, config)
        # High threshold should not match dissimilar texts
        assert len(contaminated) == 0


class TestRunCurationPipeline:
    """Tests for run_curation_pipeline function."""

    def test_basic_pipeline(self) -> None:
        """Test basic curation pipeline."""
        texts = ["  Hello  ", "  Hello  ", "World"]
        pipeline = create_curation_pipeline(steps=("clean", "dedup"))
        curated, stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) < len(texts)  # Duplicates removed
        assert stats.original_size == 3

    def test_deduplication(self) -> None:
        """Test deduplication step."""
        texts = ["Hello", "Hello", "World", "World", "Test"]
        pipeline = create_curation_pipeline(steps=("dedup",))
        curated, stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) == 3
        assert stats.removed_duplicates == 2

    def test_cleaning_step(self) -> None:
        """Test cleaning step."""
        texts = ["  Hello  ", "World  "]
        pipeline = create_curation_pipeline(steps=("clean",))
        curated, _stats = run_curation_pipeline(texts, pipeline)
        assert "Hello" in curated
        assert "  " not in curated[0]

    def test_filter_step(self) -> None:
        """Test filter step."""
        texts = ["Hello", "", "  ", "World"]
        pipeline = create_curation_pipeline(steps=("filter",))
        curated, _stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) == 2

    def test_normalize_step(self) -> None:
        """Test normalize step."""
        texts = ["  Hello   World  "]
        pipeline = create_curation_pipeline(steps=("normalize",))
        curated, _stats = run_curation_pipeline(texts, pipeline)
        assert curated[0] == "Hello World"

    def test_validate_step(self) -> None:
        """Test validate step."""
        texts = ["Hello", "   ", "World"]
        pipeline = create_curation_pipeline(steps=("validate",))
        curated, _stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) == 2

    def test_min_length_filter(self) -> None:
        """Test min_length filter in cleaning."""
        texts = ["Hi", "Hello World"]
        cleaning = create_cleaning_config(min_length=5)
        pipeline = create_curation_pipeline(
            steps=("clean",),
            cleaning_config=cleaning,
        )
        curated, _stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) == 1
        assert "Hello World" in curated

    def test_empty_texts(self) -> None:
        """Test with empty texts list."""
        curated, stats = run_curation_pipeline([])
        assert curated == []
        assert stats.original_size == 0
        assert stats.final_size == 0

    def test_none_texts_raises_error(self) -> None:
        """Test that None texts raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be None"):
            run_curation_pipeline(None)  # type: ignore[arg-type]


class TestEstimateCurationTime:
    """Tests for estimate_curation_time function."""

    def test_basic_estimate(self) -> None:
        """Test basic time estimation."""
        time = estimate_curation_time(10000)
        assert time > 0

    def test_zero_samples(self) -> None:
        """Test with zero samples."""
        time = estimate_curation_time(0)
        assert time == 0.0

    def test_negative_samples_raises_error(self) -> None:
        """Test that negative samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be non-negative"):
            estimate_curation_time(-1)

    def test_larger_dataset_takes_longer(self) -> None:
        """Test that larger datasets take longer."""
        time_small = estimate_curation_time(1000)
        time_large = estimate_curation_time(10000)
        assert time_large > time_small

    def test_more_steps_take_longer(self) -> None:
        """Test that more steps take longer."""
        pipeline_short = create_curation_pipeline(steps=("dedup",))
        pipeline_long = create_curation_pipeline(
            steps=("dedup", "clean", "filter", "validate")
        )
        time_short = estimate_curation_time(10000, pipeline_short)
        time_long = estimate_curation_time(10000, pipeline_long)
        assert time_long > time_short


class TestFormatCurationStats:
    """Tests for format_curation_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = CurationStats(
            original_size=10000,
            final_size=8500,
            removed_duplicates=1000,
            removed_contamination=500,
        )
        formatted = format_curation_stats(stats)
        assert "10,000" in formatted
        assert "8,500" in formatted
        assert "1,000" in formatted
        assert "Duplicates" in formatted
        assert "Contamination" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_curation_stats(None)  # type: ignore[arg-type]

    def test_zero_original_size(self) -> None:
        """Test with zero original size."""
        stats = CurationStats(
            original_size=0,
            final_size=0,
            removed_duplicates=0,
            removed_contamination=0,
        )
        formatted = format_curation_stats(stats)
        assert "0" in formatted

    def test_retention_rate_display(self) -> None:
        """Test that retention rate is displayed."""
        stats = CurationStats(
            original_size=1000,
            final_size=850,
            removed_duplicates=100,
            removed_contamination=50,
        )
        formatted = format_curation_stats(stats)
        assert "Retention rate" in formatted
        assert "85" in formatted  # 85% retention


class TestGetRecommendedCurationConfig:
    """Tests for get_recommended_curation_config function."""

    def test_training_use_case(self) -> None:
        """Test recommendation for training use case."""
        pipeline = get_recommended_curation_config("training")
        assert CurationStep.DEDUP in pipeline.steps
        assert CurationStep.CLEAN in pipeline.steps
        assert pipeline.cleaning_config is not None

    def test_benchmark_use_case(self) -> None:
        """Test recommendation for benchmark use case."""
        pipeline = get_recommended_curation_config("benchmark")
        assert CurationStep.CLEAN in pipeline.steps
        assert pipeline.decontam_config is not None

    def test_production_use_case(self) -> None:
        """Test recommendation for production use case."""
        pipeline = get_recommended_curation_config("production")
        assert CurationStep.CLEAN in pipeline.steps
        assert CurationStep.NORMALIZE in pipeline.steps

    def test_empty_use_case_raises_error(self) -> None:
        """Test that empty use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case cannot be empty"):
            get_recommended_curation_config("")

    def test_unknown_use_case_returns_default(self) -> None:
        """Test that unknown use case returns default config."""
        pipeline = get_recommended_curation_config("unknown_case")
        assert CurationStep.CLEAN in pipeline.steps
        assert CurationStep.DEDUP in pipeline.steps

    def test_fine_tuning_alias(self) -> None:
        """Test fine-tuning alias for training."""
        pipeline = get_recommended_curation_config("fine-tuning")
        assert CurationStep.DEDUP in pipeline.steps

    def test_evaluation_alias(self) -> None:
        """Test evaluation alias for benchmark."""
        pipeline = get_recommended_curation_config("evaluation")
        assert pipeline.decontam_config is not None

    def test_deployment_alias(self) -> None:
        """Test deployment alias for production."""
        pipeline = get_recommended_curation_config("deployment")
        assert CurationStep.NORMALIZE in pipeline.steps


class TestPropertyBased:
    """Property-based tests for curation module."""

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=20)
    def test_apply_cleaning_returns_string(self, text: str) -> None:
        """Test that apply_cleaning always returns a string."""
        result = apply_cleaning(text)
        assert isinstance(result, str)

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=20)
    def test_cleaning_does_not_increase_length(self, text: str) -> None:
        """Test that cleaning with whitespace doesn't increase length."""
        config = create_cleaning_config(methods=("whitespace",))
        result = apply_cleaning(text, config)
        # Whitespace cleaning should not increase length (normalizes spaces)
        assert len(result) <= len(text)

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20)
    def test_valid_threshold_creates_valid_config(self, threshold: float) -> None:
        """Test that valid thresholds create valid configs."""
        config = create_decontamination_config(threshold=threshold)
        assert config.threshold == pytest.approx(threshold)
        validate_decontamination_config(config)

    @given(st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20)
    def test_estimate_time_non_negative(self, num_samples: int) -> None:
        """Test that estimated time is always non-negative."""
        time = estimate_curation_time(num_samples)
        assert time >= 0.0

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=20))
    @settings(max_examples=10)
    def test_pipeline_preserves_or_reduces_size(self, texts: list[str]) -> None:
        """Test that pipeline preserves or reduces text count."""
        pipeline = create_curation_pipeline(steps=("dedup",))
        curated, stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) <= len(texts)
        assert stats.final_size <= stats.original_size


class TestEdgeCases:
    """Test edge cases for curation module."""

    def test_whitespace_only_text_cleaning(self) -> None:
        """Test cleaning whitespace-only text."""
        result = apply_cleaning("   \t\n  ")
        assert result == ""

    def test_special_characters_in_text(self) -> None:
        """Test handling of special characters."""
        text = "Hello!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = apply_cleaning(text)
        assert isinstance(result, str)

    def test_unicode_text_cleaning(self) -> None:
        """Test handling of unicode text."""
        text = "Hello \u4e16\u754c \u3053\u3093\u306b\u3061\u306f"
        result = apply_cleaning(text)
        assert isinstance(result, str)

    def test_very_long_text_cleaning(self) -> None:
        """Test handling of very long text."""
        text = "Hello world. " * 1000
        result = apply_cleaning(text)
        assert isinstance(result, str)

    def test_html_with_attributes(self) -> None:
        """Test HTML cleaning with attributes."""
        config = create_cleaning_config(methods=("html",))
        result = apply_cleaning('<div class="test">Hello</div>', config)
        assert result == "Hello"

    def test_nested_html_tags(self) -> None:
        """Test cleaning nested HTML tags."""
        config = create_cleaning_config(methods=("html",))
        result = apply_cleaning("<div><p><span>Hello</span></p></div>", config)
        assert result == "Hello"

    def test_markdown_headers(self) -> None:
        """Test markdown header cleaning."""
        config = create_cleaning_config(methods=("markdown",))
        result = apply_cleaning("# Header\nText", config)
        assert "Header" in result
        assert "#" not in result

    def test_all_cleaning_methods_combined(self) -> None:
        """Test all cleaning methods combined."""
        config = create_cleaning_config(
            methods=("whitespace", "unicode", "html", "markdown")
        )
        text = "  <p>**Bold** text</p>  "
        result = apply_cleaning(text, config)
        assert "<p>" not in result
        assert "**" not in result
        assert result.strip() == result

    def test_boundary_threshold_values(self) -> None:
        """Test boundary threshold values."""
        config_zero = create_decontamination_config(threshold=0.0)
        assert config_zero.threshold == 0.0

        config_one = create_decontamination_config(threshold=1.0)
        assert config_one.threshold == 1.0

    def test_single_sample_deduplication(self) -> None:
        """Test deduplication with single sample."""
        texts = ["Hello"]
        pipeline = create_curation_pipeline(steps=("dedup",))
        curated, stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) == 1
        assert stats.removed_duplicates == 0

    def test_all_duplicates_removal(self) -> None:
        """Test removal when all samples are duplicates."""
        texts = ["Hello", "Hello", "Hello"]
        pipeline = create_curation_pipeline(steps=("dedup",))
        curated, stats = run_curation_pipeline(texts, pipeline)
        assert len(curated) == 1
        assert stats.removed_duplicates == 2

    def test_contamination_with_short_texts(self) -> None:
        """Test contamination detection with very short texts."""
        train = ["Hi", "Bye"]
        test = ["Hi"]
        config = create_decontamination_config(
            decontam_type="exact_match",
        )
        contaminated = detect_contamination(train, test, config)
        assert len(contaminated) == 1

    def test_all_curation_steps(self) -> None:
        """Test that all curation steps are accessible."""
        steps = list_curation_steps()
        assert len(steps) == len(CurationStep)

    def test_all_cleaning_methods(self) -> None:
        """Test that all cleaning methods are accessible."""
        methods = list_cleaning_methods()
        assert len(methods) == len(CleaningMethod)

    def test_all_decontamination_types(self) -> None:
        """Test that all decontamination types are accessible."""
        types = list_decontamination_types()
        assert len(types) == len(DecontaminationType)
