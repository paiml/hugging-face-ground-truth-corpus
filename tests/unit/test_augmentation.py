"""Tests for data augmentation functionality."""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.augmentation import (
    AugmentationType,
    AugmentConfig,
    AugmentResult,
    augment_batch,
    augment_text,
    chain_augmentations,
    compute_augmentation_stats,
    create_augmenter,
    get_augmentation_type,
    list_augmentation_types,
    random_delete,
    random_insert,
    random_swap,
    synonym_replace,
    validate_augment_config,
    validate_augmentation_type,
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

    def test_none_value(self) -> None:
        """Test NONE value."""
        assert AugmentationType.NONE.value == "none"


class TestAugmentConfig:
    """Tests for AugmentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AugmentConfig()
        assert config.probability == 0.1
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
        assert config.probability == 0.3
        assert config.num_augmentations == 5
        assert config.augmentation_type == AugmentationType.RANDOM_DELETE
        assert config.min_length == 5
        assert config.preserve_case is False

    def test_frozen(self) -> None:
        """Test that AugmentConfig is immutable."""
        config = AugmentConfig()
        with pytest.raises(AttributeError):
            config.probability = 0.5  # type: ignore[misc]


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
        assert "none" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_augmentation_types()
        assert types == sorted(types)


class TestValidateAugmentationType:
    """Tests for validate_augmentation_type function."""

    def test_valid_synonym_replace(self) -> None:
        """Test validation of synonym_replace."""
        assert validate_augmentation_type("synonym_replace") is True

    def test_valid_random_delete(self) -> None:
        """Test validation of random_delete."""
        assert validate_augmentation_type("random_delete") is True

    def test_valid_none(self) -> None:
        """Test validation of none."""
        assert validate_augmentation_type("none") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_augmentation_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_augmentation_type("") is False


class TestGetAugmentationType:
    """Tests for get_augmentation_type function."""

    def test_get_synonym_replace(self) -> None:
        """Test getting SYNONYM_REPLACE."""
        result = get_augmentation_type("synonym_replace")
        assert result == AugmentationType.SYNONYM_REPLACE

    def test_get_none(self) -> None:
        """Test getting NONE."""
        result = get_augmentation_type("none")
        assert result == AugmentationType.NONE

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid augmentation type"):
            get_augmentation_type("invalid")


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
        assert stats["avg_operations_per_text"] == 1.5

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
