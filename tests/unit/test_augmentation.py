"""Tests for data augmentation functionality."""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.augmentation import (
    VALID_AUGMENTATION_LEVELS,
    VALID_AUGMENTATION_TYPES,
    VALID_NOISE_TYPES,
    AugmentationConfig,
    AugmentationLevel,
    AugmentationStats,
    AugmentationType,
    AugmentConfig,
    AugmentResult,
    BacktranslationConfig,
    NoiseConfig,
    NoiseType,
    SynonymConfig,
    apply_augmentation,
    augment_batch,
    augment_text,
    calculate_augmentation_factor,
    chain_augmentations,
    compute_augmentation_stats,
    create_augmentation_config,
    create_augmenter,
    create_backtranslation_config,
    create_noise_config,
    create_synonym_config,
    estimate_diversity_gain,
    format_augmentation_stats,
    get_augmentation_level,
    get_augmentation_type,
    get_noise_type,
    get_recommended_augmentation_config,
    inject_noise,
    list_augmentation_levels,
    list_augmentation_types,
    list_noise_types,
    random_delete,
    random_insert,
    random_swap,
    synonym_replace,
    validate_augment_config,
    validate_augmentation_config,
    validate_augmentation_level,
    validate_augmentation_type,
    validate_backtranslation_config,
    validate_noise_config,
    validate_noise_type,
    validate_synonym_config,
)


class TestAugmentationType:
    """Tests for AugmentationType enum."""

    def test_synonym_replace_value(self) -> None:
        """Test SYNONYM_REPLACE value."""
        assert AugmentationType.SYNONYM_REPLACE.value == "synonym_replace"

    def test_random_insert_value(self) -> None:
        """Test RANDOM_INSERT value."""
        assert AugmentationType.RANDOM_INSERT.value == "random_insert"

    def test_random_swap_value(self) -> None:
        """Test RANDOM_SWAP value."""
        assert AugmentationType.RANDOM_SWAP.value == "random_swap"

    def test_random_delete_value(self) -> None:
        """Test RANDOM_DELETE value."""
        assert AugmentationType.RANDOM_DELETE.value == "random_delete"

    def test_back_translate_value(self) -> None:
        """Test BACK_TRANSLATE value."""
        assert AugmentationType.BACK_TRANSLATE.value == "back_translate"

    def test_noise_value(self) -> None:
        """Test NOISE value."""
        assert AugmentationType.NOISE.value == "noise"

    def test_none_value(self) -> None:
        """Test NONE value."""
        assert AugmentationType.NONE.value == "none"


class TestNoiseType:
    """Tests for NoiseType enum."""

    def test_character_value(self) -> None:
        """Test CHARACTER value."""
        assert NoiseType.CHARACTER.value == "character"

    def test_word_value(self) -> None:
        """Test WORD value."""
        assert NoiseType.WORD.value == "word"

    def test_keyboard_value(self) -> None:
        """Test KEYBOARD value."""
        assert NoiseType.KEYBOARD.value == "keyboard"

    def test_ocr_value(self) -> None:
        """Test OCR value."""
        assert NoiseType.OCR.value == "ocr"


class TestAugmentationLevel:
    """Tests for AugmentationLevel enum."""

    def test_light_value(self) -> None:
        """Test LIGHT value."""
        assert AugmentationLevel.LIGHT.value == "light"

    def test_medium_value(self) -> None:
        """Test MEDIUM value."""
        assert AugmentationLevel.MEDIUM.value == "medium"

    def test_heavy_value(self) -> None:
        """Test HEAVY value."""
        assert AugmentationLevel.HEAVY.value == "heavy"


class TestAugmentConfig:
    """Tests for AugmentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AugmentConfig()
        assert config.probability == pytest.approx(0.1)
        assert config.num_augmentations == 1
        assert config.augmentation_type == AugmentationType.SYNONYM_REPLACE
        assert config.min_length == 3
        assert config.preserve_case is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AugmentConfig(
            probability=0.3,
            num_augmentations=5,
            augmentation_type=AugmentationType.RANDOM_DELETE,
            min_length=5,
            preserve_case=False,
        )
        assert config.probability == pytest.approx(0.3)
        assert config.num_augmentations == 5
        assert config.augmentation_type == AugmentationType.RANDOM_DELETE
        assert config.min_length == 5
        assert config.preserve_case is False

    def test_frozen(self) -> None:
        """Test that AugmentConfig is immutable."""
        config = AugmentConfig()
        with pytest.raises(AttributeError):
            config.probability = 0.5  # type: ignore[misc]


class TestSynonymConfig:
    """Tests for SynonymConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating SynonymConfig instance."""
        config = SynonymConfig(
            method="wordnet",
            max_replacements=5,
            similarity_threshold=0.8,
        )
        assert config.method == "wordnet"
        assert config.max_replacements == 5
        assert config.similarity_threshold == pytest.approx(0.8)

    def test_frozen(self) -> None:
        """Test that SynonymConfig is immutable."""
        config = SynonymConfig(
            method="wordnet",
            max_replacements=5,
            similarity_threshold=0.8,
        )
        with pytest.raises(AttributeError):
            config.method = "embedding"  # type: ignore[misc]


class TestBacktranslationConfig:
    """Tests for BacktranslationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating BacktranslationConfig instance."""
        config = BacktranslationConfig(
            pivot_languages=("de", "fr", "es"),
            num_translations=2,
        )
        assert len(config.pivot_languages) == 3
        assert config.num_translations == 2

    def test_frozen(self) -> None:
        """Test that BacktranslationConfig is immutable."""
        config = BacktranslationConfig(
            pivot_languages=("de",),
            num_translations=1,
        )
        with pytest.raises(AttributeError):
            config.num_translations = 5  # type: ignore[misc]


class TestNoiseConfig:
    """Tests for NoiseConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating NoiseConfig instance."""
        config = NoiseConfig(
            noise_type=NoiseType.CHARACTER,
            probability=0.1,
            preserve_semantics=True,
        )
        assert config.noise_type == NoiseType.CHARACTER
        assert config.probability == pytest.approx(0.1)
        assert config.preserve_semantics is True

    def test_frozen(self) -> None:
        """Test that NoiseConfig is immutable."""
        config = NoiseConfig(
            noise_type=NoiseType.KEYBOARD,
            probability=0.2,
            preserve_semantics=False,
        )
        with pytest.raises(AttributeError):
            config.probability = 0.5  # type: ignore[misc]


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating AugmentationConfig instance."""
        config = AugmentationConfig(
            aug_type=AugmentationType.SYNONYM_REPLACE,
            probability=0.3,
            preserve_labels=True,
        )
        assert config.aug_type == AugmentationType.SYNONYM_REPLACE
        assert config.probability == pytest.approx(0.3)
        assert config.preserve_labels is True

    def test_frozen(self) -> None:
        """Test that AugmentationConfig is immutable."""
        config = AugmentationConfig(
            aug_type=AugmentationType.NONE,
            probability=0.1,
            preserve_labels=False,
        )
        with pytest.raises(AttributeError):
            config.preserve_labels = True  # type: ignore[misc]


class TestAugmentResult:
    """Tests for AugmentResult dataclass."""

    def test_creation(self) -> None:
        """Test creating AugmentResult instance."""
        result = AugmentResult(
            original="hello world",
            augmented=["hi world", "hello earth"],
            operations_applied=[1, 1],
        )
        assert result.original == "hello world"
        assert len(result.augmented) == 2
        assert result.operations_applied == [1, 1]

    def test_frozen(self) -> None:
        """Test that AugmentResult is immutable."""
        result = AugmentResult(
            original="test",
            augmented=["test"],
            operations_applied=[0],
        )
        with pytest.raises(AttributeError):
            result.original = "new"  # type: ignore[misc]


class TestAugmentationStats:
    """Tests for AugmentationStats dataclass."""

    def test_creation(self) -> None:
        """Test creating AugmentationStats instance."""
        stats = AugmentationStats(
            total_texts=100,
            total_augmented=300,
            total_operations=450,
            avg_operations_per_text=4.5,
            diversity_score=0.75,
        )
        assert stats.total_texts == 100
        assert stats.total_augmented == 300
        assert stats.total_operations == 450
        assert stats.avg_operations_per_text == pytest.approx(4.5)
        assert stats.diversity_score == pytest.approx(0.75)

    def test_frozen(self) -> None:
        """Test that AugmentationStats is immutable."""
        stats = AugmentationStats(
            total_texts=10,
            total_augmented=30,
            total_operations=45,
            avg_operations_per_text=4.5,
            diversity_score=0.5,
        )
        with pytest.raises(AttributeError):
            stats.total_texts = 200  # type: ignore[misc]


class TestValidateAugmentConfig:
    """Tests for validate_augment_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = AugmentConfig(probability=0.2)
        validate_augment_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_augment_config(None)  # type: ignore[arg-type]

    def test_probability_below_zero_raises_error(self) -> None:
        """Test that probability below 0 raises ValueError."""
        config = AugmentConfig(probability=-0.1)
        with pytest.raises(ValueError, match="probability must be between 0 and 1"):
            validate_augment_config(config)

    def test_probability_above_one_raises_error(self) -> None:
        """Test that probability above 1 raises ValueError."""
        config = AugmentConfig(probability=1.5)
        with pytest.raises(ValueError, match="probability must be between 0 and 1"):
            validate_augment_config(config)

    def test_zero_augmentations_raises_error(self) -> None:
        """Test that zero num_augmentations raises ValueError."""
        config = AugmentConfig(num_augmentations=0)
        with pytest.raises(ValueError, match="num_augmentations must be positive"):
            validate_augment_config(config)

    def test_negative_min_length_raises_error(self) -> None:
        """Test that negative min_length raises ValueError."""
        config = AugmentConfig(min_length=-1)
        with pytest.raises(ValueError, match="min_length cannot be negative"):
            validate_augment_config(config)


class TestValidateSynonymConfig:
    """Tests for validate_synonym_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = SynonymConfig(
            method="wordnet",
            max_replacements=5,
            similarity_threshold=0.8,
        )
        validate_synonym_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_synonym_config(None)  # type: ignore[arg-type]

    def test_empty_method_raises_error(self) -> None:
        """Test that empty method raises ValueError."""
        config = SynonymConfig(method="", max_replacements=5, similarity_threshold=0.8)
        with pytest.raises(ValueError, match="method cannot be empty"):
            validate_synonym_config(config)

    def test_zero_max_replacements_raises_error(self) -> None:
        """Test that zero max_replacements raises ValueError."""
        config = SynonymConfig(
            method="wordnet", max_replacements=0, similarity_threshold=0.8
        )
        with pytest.raises(ValueError, match="max_replacements must be positive"):
            validate_synonym_config(config)

    def test_invalid_similarity_threshold_raises_error(self) -> None:
        """Test that invalid similarity_threshold raises ValueError."""
        config = SynonymConfig(
            method="wordnet", max_replacements=5, similarity_threshold=1.5
        )
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            validate_synonym_config(config)


class TestValidateBacktranslationConfig:
    """Tests for validate_backtranslation_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = BacktranslationConfig(pivot_languages=("de", "fr"), num_translations=2)
        validate_backtranslation_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_backtranslation_config(None)  # type: ignore[arg-type]

    def test_empty_pivot_languages_raises_error(self) -> None:
        """Test that empty pivot_languages raises ValueError."""
        config = BacktranslationConfig(pivot_languages=(), num_translations=2)
        with pytest.raises(ValueError, match="pivot_languages cannot be empty"):
            validate_backtranslation_config(config)

    def test_zero_num_translations_raises_error(self) -> None:
        """Test that zero num_translations raises ValueError."""
        config = BacktranslationConfig(pivot_languages=("de",), num_translations=0)
        with pytest.raises(ValueError, match="num_translations must be positive"):
            validate_backtranslation_config(config)


class TestValidateNoiseConfig:
    """Tests for validate_noise_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = NoiseConfig(
            noise_type=NoiseType.CHARACTER,
            probability=0.1,
            preserve_semantics=True,
        )
        validate_noise_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_noise_config(None)  # type: ignore[arg-type]

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises ValueError."""
        config = NoiseConfig(
            noise_type=NoiseType.CHARACTER,
            probability=1.5,
            preserve_semantics=True,
        )
        with pytest.raises(ValueError, match="probability must be between"):
            validate_noise_config(config)


class TestValidateAugmentationConfig:
    """Tests for validate_augmentation_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = AugmentationConfig(
            aug_type=AugmentationType.SYNONYM_REPLACE,
            probability=0.3,
            preserve_labels=True,
        )
        validate_augmentation_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_augmentation_config(None)  # type: ignore[arg-type]

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises ValueError."""
        config = AugmentationConfig(
            aug_type=AugmentationType.NONE,
            probability=-0.1,
            preserve_labels=True,
        )
        with pytest.raises(ValueError, match="probability must be between"):
            validate_augmentation_config(config)


class TestCreateSynonymConfig:
    """Tests for create_synonym_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_synonym_config()
        assert config.method == "wordnet"
        assert config.max_replacements == 5
        assert config.similarity_threshold == pytest.approx(0.8)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_synonym_config(
            method="embedding",
            max_replacements=10,
            similarity_threshold=0.9,
        )
        assert config.method == "embedding"
        assert config.max_replacements == 10
        assert config.similarity_threshold == pytest.approx(0.9)

    def test_empty_method_raises_error(self) -> None:
        """Test that empty method raises ValueError."""
        with pytest.raises(ValueError, match="method cannot be empty"):
            create_synonym_config(method="")

    def test_zero_max_replacements_raises_error(self) -> None:
        """Test that zero max_replacements raises ValueError."""
        with pytest.raises(ValueError, match="max_replacements must be positive"):
            create_synonym_config(max_replacements=0)


class TestCreateBacktranslationConfig:
    """Tests for create_backtranslation_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_backtranslation_config()
        assert len(config.pivot_languages) == 2
        assert config.num_translations == 1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_backtranslation_config(
            pivot_languages=("es", "it", "pt"),
            num_translations=3,
        )
        assert len(config.pivot_languages) == 3
        assert config.num_translations == 3

    def test_empty_pivot_languages_raises_error(self) -> None:
        """Test that empty pivot_languages raises ValueError."""
        with pytest.raises(ValueError, match="pivot_languages cannot be empty"):
            create_backtranslation_config(pivot_languages=())

    def test_zero_translations_raises_error(self) -> None:
        """Test that zero num_translations raises ValueError."""
        with pytest.raises(ValueError, match="num_translations must be positive"):
            create_backtranslation_config(num_translations=0)


class TestCreateNoiseConfig:
    """Tests for create_noise_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_noise_config()
        assert config.noise_type == NoiseType.CHARACTER
        assert config.probability == pytest.approx(0.1)
        assert config.preserve_semantics is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_noise_config(
            noise_type="keyboard",
            probability=0.2,
            preserve_semantics=False,
        )
        assert config.noise_type == NoiseType.KEYBOARD
        assert config.probability == pytest.approx(0.2)
        assert config.preserve_semantics is False

    def test_invalid_noise_type_raises_error(self) -> None:
        """Test that invalid noise_type raises ValueError."""
        with pytest.raises(ValueError, match="noise_type must be one of"):
            create_noise_config(noise_type="invalid")

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="probability must be between"):
            create_noise_config(probability=1.5)


class TestCreateAugmentationConfig:
    """Tests for create_augmentation_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_augmentation_config()
        assert config.aug_type == AugmentationType.SYNONYM_REPLACE
        assert config.probability == pytest.approx(0.2)
        assert config.preserve_labels is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_augmentation_config(
            aug_type="random_delete",
            probability=0.3,
            preserve_labels=False,
        )
        assert config.aug_type == AugmentationType.RANDOM_DELETE
        assert config.probability == pytest.approx(0.3)
        assert config.preserve_labels is False

    def test_invalid_aug_type_raises_error(self) -> None:
        """Test that invalid aug_type raises ValueError."""
        with pytest.raises(ValueError, match="aug_type must be one of"):
            create_augmentation_config(aug_type="invalid")

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="probability must be between"):
            create_augmentation_config(probability=-0.1)


class TestListAugmentationTypes:
    """Tests for list_augmentation_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_augmentation_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_augmentation_types()
        assert "synonym_replace" in types
        assert "random_delete" in types
        assert "noise" in types
        assert "none" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_augmentation_types()
        assert types == sorted(types)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_AUGMENTATION_TYPES."""
        types = list_augmentation_types()
        assert set(types) == VALID_AUGMENTATION_TYPES


class TestListNoiseTypes:
    """Tests for list_noise_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_noise_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_noise_types()
        assert "character" in types
        assert "word" in types
        assert "keyboard" in types
        assert "ocr" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_noise_types()
        assert types == sorted(types)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_NOISE_TYPES."""
        types = list_noise_types()
        assert set(types) == VALID_NOISE_TYPES


class TestListAugmentationLevels:
    """Tests for list_augmentation_levels function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        levels = list_augmentation_levels()
        assert isinstance(levels, list)

    def test_contains_expected_levels(self) -> None:
        """Test that list contains expected levels."""
        levels = list_augmentation_levels()
        assert "light" in levels
        assert "medium" in levels
        assert "heavy" in levels

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        levels = list_augmentation_levels()
        assert levels == sorted(levels)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_AUGMENTATION_LEVELS."""
        levels = list_augmentation_levels()
        assert set(levels) == VALID_AUGMENTATION_LEVELS


class TestValidateAugmentationType:
    """Tests for validate_augmentation_type function."""

    def test_valid_synonym_replace(self) -> None:
        """Test validation of synonym_replace."""
        assert validate_augmentation_type("synonym_replace") is True

    def test_valid_random_delete(self) -> None:
        """Test validation of random_delete."""
        assert validate_augmentation_type("random_delete") is True

    def test_valid_noise(self) -> None:
        """Test validation of noise."""
        assert validate_augmentation_type("noise") is True

    def test_valid_none(self) -> None:
        """Test validation of none."""
        assert validate_augmentation_type("none") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_augmentation_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_augmentation_type("") is False


class TestValidateNoiseType:
    """Tests for validate_noise_type function."""

    def test_valid_character(self) -> None:
        """Test validation of character."""
        assert validate_noise_type("character") is True

    def test_valid_keyboard(self) -> None:
        """Test validation of keyboard."""
        assert validate_noise_type("keyboard") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_noise_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_noise_type("") is False


class TestValidateAugmentationLevel:
    """Tests for validate_augmentation_level function."""

    def test_valid_light(self) -> None:
        """Test validation of light."""
        assert validate_augmentation_level("light") is True

    def test_valid_heavy(self) -> None:
        """Test validation of heavy."""
        assert validate_augmentation_level("heavy") is True

    def test_invalid_level(self) -> None:
        """Test validation of invalid level."""
        assert validate_augmentation_level("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_augmentation_level("") is False


class TestGetAugmentationType:
    """Tests for get_augmentation_type function."""

    def test_get_synonym_replace(self) -> None:
        """Test getting SYNONYM_REPLACE."""
        result = get_augmentation_type("synonym_replace")
        assert result == AugmentationType.SYNONYM_REPLACE

    def test_get_noise(self) -> None:
        """Test getting NOISE."""
        result = get_augmentation_type("noise")
        assert result == AugmentationType.NOISE

    def test_get_none(self) -> None:
        """Test getting NONE."""
        result = get_augmentation_type("none")
        assert result == AugmentationType.NONE

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid augmentation type"):
            get_augmentation_type("invalid")


class TestGetNoiseType:
    """Tests for get_noise_type function."""

    def test_get_character(self) -> None:
        """Test getting CHARACTER."""
        result = get_noise_type("character")
        assert result == NoiseType.CHARACTER

    def test_get_keyboard(self) -> None:
        """Test getting KEYBOARD."""
        result = get_noise_type("keyboard")
        assert result == NoiseType.KEYBOARD

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid noise type"):
            get_noise_type("invalid")


class TestGetAugmentationLevel:
    """Tests for get_augmentation_level function."""

    def test_get_light(self) -> None:
        """Test getting LIGHT."""
        result = get_augmentation_level("light")
        assert result == AugmentationLevel.LIGHT

    def test_get_heavy(self) -> None:
        """Test getting HEAVY."""
        result = get_augmentation_level("heavy")
        assert result == AugmentationLevel.HEAVY

    def test_invalid_level_raises_error(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="invalid augmentation level"):
            get_augmentation_level("invalid")


class TestRandomDelete:
    """Tests for random_delete function."""

    def test_zero_probability(self) -> None:
        """Test with zero probability keeps all words."""
        rng = random.Random(42)
        result = random_delete(["hello", "world", "test"], 0.0, rng=rng)
        assert result == ["hello", "world", "test"]

    def test_full_probability(self) -> None:
        """Test with probability 1.0 deletes all words."""
        rng = random.Random(42)
        result = random_delete(["hello", "world"], 1.0, rng=rng)
        assert result == []

    def test_empty_list(self) -> None:
        """Test with empty list."""
        result = random_delete([], 0.5)
        assert result == []

    def test_none_words_raises_error(self) -> None:
        """Test that None words raises ValueError."""
        with pytest.raises(ValueError, match="words cannot be None"):
            random_delete(None, 0.5)  # type: ignore[arg-type]

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="probability must be between 0 and 1"):
            random_delete(["a"], 1.5)

    @given(st.lists(st.text(min_size=1), min_size=1, max_size=20))
    @settings(max_examples=10)
    def test_result_subset_of_input(self, words: list[str]) -> None:
        """Test that result is always a subset of input."""
        result = random_delete(words, 0.5)
        for word in result:
            assert word in words


class TestRandomSwap:
    """Tests for random_swap function."""

    def test_zero_swaps(self) -> None:
        """Test with zero swaps keeps original order."""
        result = random_swap(["a", "b", "c"], 0)
        assert result == ["a", "b", "c"]

    def test_single_word(self) -> None:
        """Test with single word returns same."""
        result = random_swap(["a"], 5)
        assert result == ["a"]

    def test_preserves_words(self) -> None:
        """Test that swapping preserves all words."""
        rng = random.Random(42)
        words = ["a", "b", "c", "d"]
        result = random_swap(words, 3, rng=rng)
        assert sorted(result) == sorted(words)

    def test_none_words_raises_error(self) -> None:
        """Test that None words raises ValueError."""
        with pytest.raises(ValueError, match="words cannot be None"):
            random_swap(None, 1)  # type: ignore[arg-type]

    def test_negative_swaps_raises_error(self) -> None:
        """Test that negative num_swaps raises ValueError."""
        with pytest.raises(ValueError, match="num_swaps cannot be negative"):
            random_swap(["a", "b"], -1)


class TestRandomInsert:
    """Tests for random_insert function."""

    def test_basic_insert(self) -> None:
        """Test basic insertion."""
        rng = random.Random(42)
        result = random_insert(["hello", "world"], ["new"], 1, rng=rng)
        assert len(result) == 3
        assert "new" in result

    def test_empty_pool(self) -> None:
        """Test with empty word pool."""
        result = random_insert(["a"], [], 5)
        assert result == ["a"]

    def test_zero_inserts(self) -> None:
        """Test with zero insertions."""
        result = random_insert(["a", "b"], ["x"], 0)
        assert result == ["a", "b"]

    def test_none_words_raises_error(self) -> None:
        """Test that None words raises ValueError."""
        with pytest.raises(ValueError, match="words cannot be None"):
            random_insert(None, ["x"], 1)  # type: ignore[arg-type]

    def test_none_pool_raises_error(self) -> None:
        """Test that None word_pool raises ValueError."""
        with pytest.raises(ValueError, match="word_pool cannot be None"):
            random_insert(["a"], None, 1)  # type: ignore[arg-type]

    def test_negative_inserts_raises_error(self) -> None:
        """Test that negative num_inserts raises ValueError."""
        with pytest.raises(ValueError, match="num_inserts cannot be negative"):
            random_insert(["a"], ["x"], -1)


class TestSynonymReplace:
    """Tests for synonym_replace function."""

    def test_basic_replacement(self) -> None:
        """Test basic synonym replacement."""
        synonyms = {"hello": ["hi", "hey"]}
        rng = random.Random(42)
        result = synonym_replace(["hello", "world"], synonyms, 1.0, rng=rng)
        assert result[0] in ["hi", "hey"]
        assert result[1] == "world"

    def test_empty_synonyms(self) -> None:
        """Test with empty synonyms dictionary."""
        result = synonym_replace(["a", "b"], {}, 1.0)
        assert result == ["a", "b"]

    def test_zero_probability(self) -> None:
        """Test with zero probability."""
        synonyms = {"a": ["x"]}
        result = synonym_replace(["a", "b"], synonyms, 0.0)
        assert result == ["a", "b"]

    def test_preserves_case(self) -> None:
        """Test that case is preserved."""
        synonyms = {"hello": ["hi"]}
        rng = random.Random(42)
        result = synonym_replace(["Hello"], synonyms, 1.0, rng=rng)
        assert result[0] == "Hi"

    def test_none_words_raises_error(self) -> None:
        """Test that None words raises ValueError."""
        with pytest.raises(ValueError, match="words cannot be None"):
            synonym_replace(None, {}, 0.5)  # type: ignore[arg-type]

    def test_none_synonyms_raises_error(self) -> None:
        """Test that None synonyms raises ValueError."""
        with pytest.raises(ValueError, match="synonyms cannot be None"):
            synonym_replace(["a"], None, 0.5)  # type: ignore[arg-type]

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="probability must be between 0 and 1"):
            synonym_replace(["a"], {}, 1.5)


class TestInjectNoise:
    """Tests for inject_noise function."""

    def test_zero_probability(self) -> None:
        """Test that zero probability returns original."""
        config = NoiseConfig(
            noise_type=NoiseType.CHARACTER,
            probability=0.0,
            preserve_semantics=True,
        )
        result = inject_noise("hello world", config)
        assert result == "hello world"

    def test_empty_text(self) -> None:
        """Test with empty text."""
        config = NoiseConfig(
            noise_type=NoiseType.CHARACTER,
            probability=0.5,
            preserve_semantics=True,
        )
        result = inject_noise("", config)
        assert result == ""

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        config = NoiseConfig(
            noise_type=NoiseType.CHARACTER,
            probability=0.1,
            preserve_semantics=True,
        )
        with pytest.raises(ValueError, match="text cannot be None"):
            inject_noise(None, config)  # type: ignore[arg-type]

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            inject_noise("hello", None)  # type: ignore[arg-type]

    def test_character_noise(self) -> None:
        """Test character-level noise injection."""
        config = NoiseConfig(
            noise_type=NoiseType.CHARACTER,
            probability=1.0,
            preserve_semantics=False,
        )
        rng = random.Random(42)
        result = inject_noise("hello", config, rng=rng)
        # With probability 1.0, all chars should be changed
        assert result != "hello"

    def test_word_noise(self) -> None:
        """Test word-level noise injection."""
        config = NoiseConfig(
            noise_type=NoiseType.WORD,
            probability=1.0,
            preserve_semantics=False,
        )
        rng = random.Random(42)
        result = inject_noise("hello world test example", config, rng=rng)
        # Text may be modified (words dropped or shuffled)
        assert isinstance(result, str)

    def test_keyboard_noise(self) -> None:
        """Test keyboard-based noise injection."""
        config = NoiseConfig(
            noise_type=NoiseType.KEYBOARD,
            probability=1.0,
            preserve_semantics=False,
        )
        rng = random.Random(42)
        result = inject_noise("hello", config, rng=rng)
        # Keyboard noise may change characters to nearby keys
        assert isinstance(result, str)

    def test_ocr_noise(self) -> None:
        """Test OCR-like noise injection."""
        config = NoiseConfig(
            noise_type=NoiseType.OCR,
            probability=1.0,
            preserve_semantics=False,
        )
        rng = random.Random(42)
        # Use text with characters that have OCR confusions
        result = inject_noise("hello 0 1", config, rng=rng)
        assert isinstance(result, str)


class TestAugmentText:
    """Tests for augment_text function."""

    def test_none_augmentation(self) -> None:
        """Test with NONE augmentation type."""
        config = AugmentConfig(augmentation_type=AugmentationType.NONE)
        result = augment_text("hello world", config)
        assert result.original == "hello world"
        assert result.augmented == ["hello world"]
        assert result.operations_applied == [0]

    def test_short_text_skipped(self) -> None:
        """Test that short text is not augmented."""
        config = AugmentConfig(min_length=5)
        result = augment_text("hi", config)
        assert result.augmented == ["hi"]

    def test_random_delete_augmentation(self) -> None:
        """Test RANDOM_DELETE augmentation."""
        config = AugmentConfig(
            augmentation_type=AugmentationType.RANDOM_DELETE,
            probability=0.5,
        )
        rng = random.Random(42)
        result = augment_text("hello world test example", config, rng=rng)
        assert len(result.augmented[0].split()) <= 4

    def test_random_swap_augmentation(self) -> None:
        """Test RANDOM_SWAP augmentation."""
        config = AugmentConfig(
            augmentation_type=AugmentationType.RANDOM_SWAP,
            probability=0.5,
        )
        rng = random.Random(42)
        result = augment_text("a b c d e", config, rng=rng)
        assert sorted(result.augmented[0].split()) == ["a", "b", "c", "d", "e"]

    def test_noise_augmentation(self) -> None:
        """Test NOISE augmentation."""
        config = AugmentConfig(
            augmentation_type=AugmentationType.NOISE,
            probability=0.5,
        )
        rng = random.Random(42)
        result = augment_text("hello world test", config, rng=rng)
        assert isinstance(result.augmented[0], str)

    def test_multiple_augmentations(self) -> None:
        """Test generating multiple augmented versions."""
        config = AugmentConfig(
            augmentation_type=AugmentationType.NONE,
            num_augmentations=3,
        )
        result = augment_text("hello world test", config)
        assert len(result.augmented) == 3

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            augment_text(None)  # type: ignore[arg-type]

    def test_default_config(self) -> None:
        """Test with default config."""
        result = augment_text("hello world test example")
        assert result.original == "hello world test example"


class TestApplyAugmentation:
    """Tests for apply_augmentation function."""

    def test_none_type(self) -> None:
        """Test with NONE augmentation type."""
        config = AugmentationConfig(
            aug_type=AugmentationType.NONE,
            probability=0.2,
            preserve_labels=True,
        )
        result = apply_augmentation("hello world", config)
        assert result == "hello world"

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        config = AugmentationConfig(
            aug_type=AugmentationType.NONE,
            probability=0.2,
            preserve_labels=True,
        )
        with pytest.raises(ValueError, match="text cannot be None"):
            apply_augmentation(None, config)  # type: ignore[arg-type]

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            apply_augmentation("hello", None)  # type: ignore[arg-type]

    def test_probabilistic_application(self) -> None:
        """Test that augmentation respects probability."""
        config = AugmentationConfig(
            aug_type=AugmentationType.RANDOM_DELETE,
            probability=0.0,  # 0% chance to apply
            preserve_labels=True,
        )
        rng = random.Random(42)
        result = apply_augmentation("hello world test", config, rng=rng)
        # With 0% probability, text should be unchanged
        assert result == "hello world test"


class TestCreateAugmenter:
    """Tests for create_augmenter function."""

    def test_creates_callable(self) -> None:
        """Test that create_augmenter returns a callable."""
        config = AugmentConfig(augmentation_type=AugmentationType.NONE)
        augmenter = create_augmenter(config)
        assert callable(augmenter)

    def test_augmenter_works(self) -> None:
        """Test that created augmenter works correctly."""
        config = AugmentConfig(augmentation_type=AugmentationType.NONE)
        augmenter = create_augmenter(config)
        result = augmenter("hello world")
        assert result.original == "hello world"

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            create_augmenter(None)  # type: ignore[arg-type]

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        config = AugmentConfig(probability=-0.5)
        with pytest.raises(ValueError, match="probability must be between 0 and 1"):
            create_augmenter(config)


class TestAugmentBatch:
    """Tests for augment_batch function."""

    def test_basic_batch(self) -> None:
        """Test basic batch augmentation."""
        config = AugmentConfig(augmentation_type=AugmentationType.NONE)
        results = augment_batch(["hello", "world"], config)
        assert len(results) == 2
        assert results[0].original == "hello"
        assert results[1].original == "world"

    def test_empty_batch(self) -> None:
        """Test with empty batch."""
        results = augment_batch([])
        assert results == []

    def test_none_texts_raises_error(self) -> None:
        """Test that None texts raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be None"):
            augment_batch(None)  # type: ignore[arg-type]


class TestChainAugmentations:
    """Tests for chain_augmentations function."""

    def test_empty_chain(self) -> None:
        """Test with empty augmentation chain."""
        result = chain_augmentations("hello world test", [])
        assert result == "hello world test"

    def test_single_none_augmentation(self) -> None:
        """Test with single NONE augmentation."""
        result = chain_augmentations("hello world test", [AugmentationType.NONE])
        assert result == "hello world test"

    def test_multiple_augmentations(self) -> None:
        """Test with multiple augmentations."""
        rng = random.Random(42)
        result = chain_augmentations(
            "hello world test example",
            [AugmentationType.RANDOM_SWAP, AugmentationType.RANDOM_DELETE],
            probability=0.3,
            rng=rng,
        )
        assert isinstance(result, str)

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            chain_augmentations(None, [])  # type: ignore[arg-type]

    def test_none_types_raises_error(self) -> None:
        """Test that None augmentation_types raises ValueError."""
        with pytest.raises(ValueError, match="augmentation_types cannot be None"):
            chain_augmentations("test", None)  # type: ignore[arg-type]


class TestComputeAugmentationStats:
    """Tests for compute_augmentation_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic statistics computation."""
        results = [
            AugmentResult("a", ["b"], [1]),
            AugmentResult("c", ["d"], [2]),
        ]
        stats = compute_augmentation_stats(results)
        assert stats["total_texts"] == 2
        assert stats["total_augmented"] == 2
        assert stats["total_operations"] == 3
        assert stats["avg_operations_per_text"] == pytest.approx(1.5)

    def test_empty_results(self) -> None:
        """Test with empty results."""
        stats = compute_augmentation_stats([])
        assert stats["total_texts"] == 0
        assert stats["total_operations"] == 0

    def test_multiple_augmentations_per_text(self) -> None:
        """Test with multiple augmentations per text."""
        results = [
            AugmentResult("a", ["b", "c"], [1, 2]),
        ]
        stats = compute_augmentation_stats(results)
        assert stats["total_augmented"] == 2
        assert stats["total_operations"] == 3

    def test_none_results_raises_error(self) -> None:
        """Test that None results raises ValueError."""
        with pytest.raises(ValueError, match="results cannot be None"):
            compute_augmentation_stats(None)  # type: ignore[arg-type]


class TestCalculateAugmentationFactor:
    """Tests for calculate_augmentation_factor function."""

    def test_basic_factor(self) -> None:
        """Test basic factor calculation."""
        factor = calculate_augmentation_factor(3, 2)
        assert factor == pytest.approx(7.0)

    def test_zero_techniques(self) -> None:
        """Test with zero techniques."""
        factor = calculate_augmentation_factor(1, 0)
        assert factor == pytest.approx(1.0)

    def test_with_chain_probability(self) -> None:
        """Test with chain_probability."""
        factor = calculate_augmentation_factor(2, 1, chain_probability=0.5)
        assert factor == pytest.approx(2.0)

    def test_zero_augmentations_raises_error(self) -> None:
        """Test that zero num_augmentations raises ValueError."""
        with pytest.raises(ValueError, match="num_augmentations must be positive"):
            calculate_augmentation_factor(0, 1)

    def test_negative_techniques_raises_error(self) -> None:
        """Test that negative num_techniques raises ValueError."""
        with pytest.raises(ValueError, match="num_techniques cannot be negative"):
            calculate_augmentation_factor(1, -1)

    def test_invalid_chain_probability_raises_error(self) -> None:
        """Test that invalid chain_probability raises ValueError."""
        with pytest.raises(ValueError, match="chain_probability must be between"):
            calculate_augmentation_factor(1, 1, chain_probability=1.5)


class TestEstimateDiversityGain:
    """Tests for estimate_diversity_gain function."""

    def test_full_diversity(self) -> None:
        """Test with all augmented texts different from original."""
        results = [
            AugmentResult("hello", ["hi", "hey"], [1, 1]),
            AugmentResult("world", ["earth", "globe"], [1, 1]),
        ]
        score = estimate_diversity_gain(results)
        assert score == pytest.approx(1.0)

    def test_no_diversity(self) -> None:
        """Test with all augmented texts same as original."""
        results = [
            AugmentResult("hello", ["hello", "hello"], [0, 0]),
        ]
        score = estimate_diversity_gain(results)
        assert score == pytest.approx(0.0)

    def test_empty_results(self) -> None:
        """Test with empty results."""
        score = estimate_diversity_gain([])
        assert score == pytest.approx(0.0)

    def test_none_results_raises_error(self) -> None:
        """Test that None results raises ValueError."""
        with pytest.raises(ValueError, match="results cannot be None"):
            estimate_diversity_gain(None)  # type: ignore[arg-type]

    def test_partial_diversity(self) -> None:
        """Test with partial diversity."""
        results = [
            AugmentResult("hello", ["hi", "hello"], [1, 0]),
        ]
        score = estimate_diversity_gain(results)
        assert score == pytest.approx(0.5)


class TestFormatAugmentationStats:
    """Tests for format_augmentation_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = AugmentationStats(
            total_texts=100,
            total_augmented=300,
            total_operations=450,
            avg_operations_per_text=4.5,
            diversity_score=0.75,
        )
        formatted = format_augmentation_stats(stats)
        assert "100" in formatted
        assert "300" in formatted
        assert "450" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_augmentation_stats(None)  # type: ignore[arg-type]

    def test_zero_totals(self) -> None:
        """Test with zero totals."""
        stats = AugmentationStats(
            total_texts=0,
            total_augmented=0,
            total_operations=0,
            avg_operations_per_text=0.0,
            diversity_score=0.0,
        )
        formatted = format_augmentation_stats(stats)
        assert "0" in formatted


class TestGetRecommendedAugmentationConfig:
    """Tests for get_recommended_augmentation_config function."""

    def test_classification_task(self) -> None:
        """Test recommendation for classification task."""
        config = get_recommended_augmentation_config("classification", 1000)
        assert config.preserve_labels is True
        assert config.aug_type == AugmentationType.SYNONYM_REPLACE

    def test_ner_task(self) -> None:
        """Test recommendation for NER task."""
        config = get_recommended_augmentation_config("ner", 1000)
        assert config.preserve_labels is True
        assert config.aug_type == AugmentationType.RANDOM_SWAP

    def test_qa_task(self) -> None:
        """Test recommendation for QA task."""
        config = get_recommended_augmentation_config("qa", 1000)
        assert config.aug_type == AugmentationType.BACK_TRANSLATE

    def test_generation_task(self) -> None:
        """Test recommendation for generation task."""
        config = get_recommended_augmentation_config("generation", 100000)
        assert config.preserve_labels is False

    def test_small_dataset(self) -> None:
        """Test recommendation for small dataset."""
        config = get_recommended_augmentation_config("classification", 500)
        assert config.probability == pytest.approx(0.3)

    def test_large_dataset(self) -> None:
        """Test recommendation for large dataset."""
        config = get_recommended_augmentation_config("classification", 100000)
        assert config.probability == pytest.approx(0.1)

    def test_empty_task_type_raises_error(self) -> None:
        """Test that empty task_type raises ValueError."""
        with pytest.raises(ValueError, match="task_type cannot be empty"):
            get_recommended_augmentation_config("", 1000)

    def test_zero_dataset_size_raises_error(self) -> None:
        """Test that zero dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            get_recommended_augmentation_config("classification", 0)

    def test_negative_dataset_size_raises_error(self) -> None:
        """Test that negative dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            get_recommended_augmentation_config("classification", -1)


class TestRandomInsertAugmentation:
    """Tests for RANDOM_INSERT augmentation via augment_text."""

    def test_random_insert_augmentation(self) -> None:
        """Test RANDOM_INSERT augmentation type."""
        config = AugmentConfig(
            augmentation_type=AugmentationType.RANDOM_INSERT,
            probability=0.5,
        )
        rng = random.Random(42)
        result = augment_text("hello world test", config, rng=rng)
        assert len(result.augmented[0].split()) >= 3


class TestSynonymReplaceAugmentation:
    """Tests for SYNONYM_REPLACE augmentation via augment_text."""

    def test_with_synonyms(self) -> None:
        """Test synonym replacement with synonyms dictionary."""
        config = AugmentConfig(
            augmentation_type=AugmentationType.SYNONYM_REPLACE,
            probability=1.0,
        )
        synonyms = {"hello": ["hi"], "world": ["earth"]}
        rng = random.Random(42)
        result = augment_text("hello world test", config, synonyms=synonyms, rng=rng)
        # At least one replacement should occur
        ops = result.operations_applied[0]
        assert result.augmented[0] != "hello world test" or ops == 0

    def test_without_synonyms(self) -> None:
        """Test synonym replacement without synonyms dictionary."""
        config = AugmentConfig(
            augmentation_type=AugmentationType.SYNONYM_REPLACE,
            probability=1.0,
        )
        result = augment_text("hello world test", config)
        # Without synonyms, text should remain unchanged
        assert result.augmented[0] == "hello world test"


class TestPropertyBased:
    """Property-based tests for augmentation module."""

    @given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
    @settings(max_examples=10)
    def test_random_delete_preserves_subset(self, words: list[str]) -> None:
        """Test that random_delete result is subset of input."""
        result = random_delete(words, 0.5)
        for word in result:
            assert word in words

    @given(st.lists(st.text(min_size=1), min_size=2, max_size=10))
    @settings(max_examples=10)
    def test_random_swap_preserves_words(self, words: list[str]) -> None:
        """Test that random_swap preserves all words."""
        result = random_swap(words, 2)
        assert sorted(result) == sorted(words)

    @given(
        st.integers(min_value=1, max_value=10), st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=10)
    def test_augmentation_factor_positive(self, num_augs: int, num_techs: int) -> None:
        """Test that augmentation factor is always positive."""
        factor = calculate_augmentation_factor(num_augs, num_techs)
        assert factor >= 1.0

    @given(st.text(min_size=4, max_size=50))
    @settings(max_examples=10)
    def test_augment_text_returns_result(self, text: str) -> None:
        """Test that augment_text always returns a result."""
        config = AugmentConfig(augmentation_type=AugmentationType.NONE)
        result = augment_text(text, config)
        assert result.original == text
        assert len(result.augmented) > 0


class TestValidFrozensets:
    """Tests for VALID_* frozensets."""

    def test_valid_augmentation_types_is_frozenset(self) -> None:
        """Test that VALID_AUGMENTATION_TYPES is a frozenset."""
        assert isinstance(VALID_AUGMENTATION_TYPES, frozenset)
        assert len(VALID_AUGMENTATION_TYPES) == 7

    def test_valid_noise_types_is_frozenset(self) -> None:
        """Test that VALID_NOISE_TYPES is a frozenset."""
        assert isinstance(VALID_NOISE_TYPES, frozenset)
        assert len(VALID_NOISE_TYPES) == 4

    def test_valid_augmentation_levels_is_frozenset(self) -> None:
        """Test that VALID_AUGMENTATION_LEVELS is a frozenset."""
        assert isinstance(VALID_AUGMENTATION_LEVELS, frozenset)
        assert len(VALID_AUGMENTATION_LEVELS) == 3
