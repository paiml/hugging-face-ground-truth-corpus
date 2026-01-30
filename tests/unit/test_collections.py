"""Tests for hub.collections module."""

from __future__ import annotations

import pytest

from hf_gtc.hub.collections import (
    VALID_ITEM_TYPES,
    VALID_SORT_OPTIONS,
    VALID_THEMES,
    CollectionConfig,
    CollectionItem,
    CollectionItemType,
    CollectionMetadata,
    CollectionQuery,
    CollectionStats,
    CollectionTheme,
    calculate_collection_score,
    create_collection_config,
    create_collection_item,
    create_collection_query,
    create_collection_stats,
    format_collection_slug,
    get_item_type,
    list_item_types,
    list_sort_options,
    list_themes,
    validate_collection_config,
    validate_collection_item,
)


class TestCollectionItemType:
    """Tests for CollectionItemType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for item_type in CollectionItemType:
            assert isinstance(item_type.value, str)

    def test_model_value(self) -> None:
        """Model has correct value."""
        assert CollectionItemType.MODEL.value == "model"

    def test_dataset_value(self) -> None:
        """Dataset has correct value."""
        assert CollectionItemType.DATASET.value == "dataset"

    def test_valid_item_types_frozenset(self) -> None:
        """VALID_ITEM_TYPES is a frozenset."""
        assert isinstance(VALID_ITEM_TYPES, frozenset)


class TestCollectionTheme:
    """Tests for CollectionTheme enum."""

    def test_all_themes_have_values(self) -> None:
        """All themes have string values."""
        for theme in CollectionTheme:
            assert isinstance(theme.value, str)

    def test_grid_value(self) -> None:
        """Grid has correct value."""
        assert CollectionTheme.GRID.value == "grid"

    def test_valid_themes_frozenset(self) -> None:
        """VALID_THEMES is a frozenset."""
        assert isinstance(VALID_THEMES, frozenset)


class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create collection config."""
        config = CollectionConfig(
            title="My Collection",
            description="Description",
            private=False,
            theme=CollectionTheme.DEFAULT,
            namespace="username",
        )
        assert config.title == "My Collection"

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = CollectionConfig(
            "Title", "Desc", False, CollectionTheme.DEFAULT, "user"
        )
        with pytest.raises(AttributeError):
            config.title = "New Title"  # type: ignore[misc]


class TestValidateCollectionConfig:
    """Tests for validate_collection_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = CollectionConfig(
            "Title", "Desc", False, CollectionTheme.DEFAULT, "user"
        )
        validate_collection_config(config)

    def test_empty_title_raises(self) -> None:
        """Empty title raises ValueError."""
        config = CollectionConfig("", "Desc", False, CollectionTheme.DEFAULT, "user")
        with pytest.raises(ValueError, match="title cannot be empty"):
            validate_collection_config(config)

    def test_empty_namespace_raises(self) -> None:
        """Empty namespace raises ValueError."""
        config = CollectionConfig("Title", "Desc", False, CollectionTheme.DEFAULT, "")
        with pytest.raises(ValueError, match="namespace cannot be empty"):
            validate_collection_config(config)

    def test_title_too_long_raises(self) -> None:
        """Title too long raises ValueError."""
        config = CollectionConfig(
            "X" * 101, "Desc", False, CollectionTheme.DEFAULT, "user"
        )
        with pytest.raises(ValueError, match="title cannot exceed 100 characters"):
            validate_collection_config(config)


class TestCollectionItem:
    """Tests for CollectionItem dataclass."""

    def test_create_item(self) -> None:
        """Create collection item."""
        item = CollectionItem(
            item_id="user/repo",
            item_type=CollectionItemType.MODEL,
            note="Great model",
            position=0,
        )
        assert item.item_id == "user/repo"


class TestValidateCollectionItem:
    """Tests for validate_collection_item function."""

    def test_valid_item(self) -> None:
        """Valid item passes validation."""
        item = CollectionItem("user/repo", CollectionItemType.MODEL, "", 0)
        validate_collection_item(item)

    def test_empty_item_id_raises(self) -> None:
        """Empty item_id raises ValueError."""
        item = CollectionItem("", CollectionItemType.MODEL, "", 0)
        with pytest.raises(ValueError, match="item_id cannot be empty"):
            validate_collection_item(item)

    def test_invalid_item_id_format_raises(self) -> None:
        """Invalid item_id format raises ValueError."""
        item = CollectionItem("invalid", CollectionItemType.MODEL, "", 0)
        with pytest.raises(ValueError, match="must be in 'owner/name' format"):
            validate_collection_item(item)

    def test_negative_position_raises(self) -> None:
        """Negative position raises ValueError."""
        item = CollectionItem("user/repo", CollectionItemType.MODEL, "", -1)
        with pytest.raises(ValueError, match="position must be non-negative"):
            validate_collection_item(item)


class TestCreateCollectionConfig:
    """Tests for create_collection_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_collection_config("My LLMs", namespace="username")
        assert config.title == "My LLMs"
        assert config.private is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_collection_config(
            "My LLMs",
            private=True,
            theme="grid",
            namespace="user",
        )
        assert config.private is True
        assert config.theme == CollectionTheme.GRID

    def test_empty_title_raises(self) -> None:
        """Empty title raises ValueError."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            create_collection_config("")

    def test_invalid_theme_raises(self) -> None:
        """Invalid theme raises ValueError."""
        with pytest.raises(ValueError, match="theme must be one of"):
            create_collection_config("Title", theme="invalid", namespace="user")


class TestCreateCollectionItem:
    """Tests for create_collection_item function."""

    def test_default_item(self) -> None:
        """Create default item."""
        item = create_collection_item("meta-llama/Llama-2-7b")
        assert item.item_id == "meta-llama/Llama-2-7b"
        assert item.item_type == CollectionItemType.MODEL

    def test_custom_item(self) -> None:
        """Create custom item."""
        item = create_collection_item(
            "user/dataset",
            item_type="dataset",
            note="Great dataset",
        )
        assert item.item_type == CollectionItemType.DATASET
        assert item.note == "Great dataset"

    def test_empty_item_id_raises(self) -> None:
        """Empty item_id raises ValueError."""
        with pytest.raises(ValueError, match="item_id cannot be empty"):
            create_collection_item("")

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="item_type must be one of"):
            create_collection_item("user/repo", item_type="invalid")


class TestCreateCollectionQuery:
    """Tests for create_collection_query function."""

    def test_default_query(self) -> None:
        """Create default query."""
        query = create_collection_query()
        assert query.sort == "likes"
        assert query.limit == 20

    def test_custom_query(self) -> None:
        """Create custom query."""
        query = create_collection_query(owner="huggingface", limit=10)
        assert query.owner == "huggingface"
        assert query.limit == 10

    def test_invalid_sort_raises(self) -> None:
        """Invalid sort raises ValueError."""
        with pytest.raises(ValueError, match="sort must be one of"):
            create_collection_query(sort="invalid")  # type: ignore[arg-type]

    def test_zero_limit_raises(self) -> None:
        """Zero limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be positive"):
            create_collection_query(limit=0)

    def test_limit_too_large_raises(self) -> None:
        """Limit too large raises ValueError."""
        with pytest.raises(ValueError, match="limit cannot exceed 100"):
            create_collection_query(limit=101)


class TestCreateCollectionStats:
    """Tests for create_collection_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_collection_stats()
        assert stats.total_models == 0
        assert stats.total_likes == 0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_collection_stats(total_models=5, total_likes=100)
        assert stats.total_models == 5
        assert stats.total_likes == 100

    def test_negative_models_raises(self) -> None:
        """Negative models raises ValueError."""
        with pytest.raises(ValueError, match="total_models must be non-negative"):
            create_collection_stats(total_models=-1)

    def test_negative_datasets_raises(self) -> None:
        """Negative datasets raises ValueError."""
        with pytest.raises(ValueError, match="total_datasets must be non-negative"):
            create_collection_stats(total_datasets=-1)

    def test_negative_spaces_raises(self) -> None:
        """Negative spaces raises ValueError."""
        with pytest.raises(ValueError, match="total_spaces must be non-negative"):
            create_collection_stats(total_spaces=-1)

    def test_negative_papers_raises(self) -> None:
        """Negative papers raises ValueError."""
        with pytest.raises(ValueError, match="total_papers must be non-negative"):
            create_collection_stats(total_papers=-1)

    def test_negative_likes_raises(self) -> None:
        """Negative likes raises ValueError."""
        with pytest.raises(ValueError, match="total_likes must be non-negative"):
            create_collection_stats(total_likes=-1)

    def test_negative_downloads_raises(self) -> None:
        """Negative downloads raises ValueError."""
        with pytest.raises(ValueError, match="total_downloads must be non-negative"):
            create_collection_stats(total_downloads=-1)


class TestListItemTypes:
    """Tests for list_item_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_item_types()
        assert types == sorted(types)

    def test_contains_model(self) -> None:
        """Contains model."""
        types = list_item_types()
        assert "model" in types

    def test_contains_dataset(self) -> None:
        """Contains dataset."""
        types = list_item_types()
        assert "dataset" in types


class TestListThemes:
    """Tests for list_themes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        themes = list_themes()
        assert themes == sorted(themes)

    def test_contains_default(self) -> None:
        """Contains default."""
        themes = list_themes()
        assert "default" in themes


class TestListSortOptions:
    """Tests for list_sort_options function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        options = list_sort_options()
        assert options == sorted(options)

    def test_contains_likes(self) -> None:
        """Contains likes."""
        options = list_sort_options()
        assert "likes" in options


class TestGetItemType:
    """Tests for get_item_type function."""

    def test_get_model(self) -> None:
        """Get model type."""
        assert get_item_type("model") == CollectionItemType.MODEL

    def test_get_dataset(self) -> None:
        """Get dataset type."""
        assert get_item_type("dataset") == CollectionItemType.DATASET

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="item type must be one of"):
            get_item_type("invalid")


class TestFormatCollectionSlug:
    """Tests for format_collection_slug function."""

    def test_basic_slug(self) -> None:
        """Format basic slug."""
        slug = format_collection_slug("username", "My Collection", "abc123")
        assert slug == "username/my-collection-abc123"

    def test_empty_namespace_raises(self) -> None:
        """Empty namespace raises ValueError."""
        with pytest.raises(ValueError, match="namespace cannot be empty"):
            format_collection_slug("", "Title", "id")

    def test_empty_title_raises(self) -> None:
        """Empty title raises ValueError."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            format_collection_slug("user", "", "id")

    def test_empty_id_raises(self) -> None:
        """Empty unique_id raises ValueError."""
        with pytest.raises(ValueError, match="unique_id cannot be empty"):
            format_collection_slug("user", "Title", "")


class TestCalculateCollectionScore:
    """Tests for calculate_collection_score function."""

    def test_basic_score(self) -> None:
        """Calculate basic score."""
        stats = create_collection_stats(total_models=10, total_likes=100)
        score = calculate_collection_score(stats)
        assert score > 0

    def test_higher_models_higher_score(self) -> None:
        """More models means higher score."""
        stats1 = create_collection_stats(total_models=5)
        stats2 = create_collection_stats(total_models=10)
        assert calculate_collection_score(stats2) > calculate_collection_score(stats1)


class TestCollectionMetadata:
    """Tests for CollectionMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Create collection metadata."""
        meta = CollectionMetadata(
            slug="user/my-collection-abc123",
            item_count=10,
            likes=50,
            is_featured=False,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-06-01T00:00:00Z",
        )
        assert meta.item_count == 10


class TestCollectionQuery:
    """Tests for CollectionQuery dataclass."""

    def test_create_query(self) -> None:
        """Create collection query."""
        query = CollectionQuery(
            owner="huggingface",
            item=None,
            sort="likes",
            limit=10,
        )
        assert query.owner == "huggingface"


class TestCollectionStats:
    """Tests for CollectionStats dataclass."""

    def test_create_stats(self) -> None:
        """Create collection stats."""
        stats = CollectionStats(
            total_models=5,
            total_datasets=3,
            total_spaces=2,
            total_papers=1,
            total_likes=1000,
            total_downloads=50000,
        )
        assert stats.total_models == 5


class TestValidSortOptions:
    """Tests for VALID_SORT_OPTIONS constant."""

    def test_valid_sort_options_frozenset(self) -> None:
        """VALID_SORT_OPTIONS is a frozenset."""
        assert isinstance(VALID_SORT_OPTIONS, frozenset)

    def test_contains_expected_values(self) -> None:
        """Contains expected values."""
        assert "likes" in VALID_SORT_OPTIONS
        assert "downloads" in VALID_SORT_OPTIONS
