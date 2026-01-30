"""HuggingFace Collections API utilities.

This module provides functions for working with HuggingFace Collections
for organizing models, datasets, and spaces.

Examples:
    >>> from hf_gtc.hub.collections import create_collection_config
    >>> config = create_collection_config(title="My Models", private=False)
    >>> config.title
    'My Models'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class CollectionItemType(Enum):
    """Supported collection item types.

    Attributes:
        MODEL: Model repository.
        DATASET: Dataset repository.
        SPACE: Space application.
        PAPER: Paper reference.

    Examples:
        >>> CollectionItemType.MODEL.value
        'model'
        >>> CollectionItemType.DATASET.value
        'dataset'
    """

    MODEL = "model"
    DATASET = "dataset"
    SPACE = "space"
    PAPER = "paper"


VALID_ITEM_TYPES = frozenset(i.value for i in CollectionItemType)


class CollectionTheme(Enum):
    """Collection display themes.

    Attributes:
        DEFAULT: Default theme.
        MINIMAL: Minimal display theme.
        GRID: Grid layout theme.
        LIST: List layout theme.

    Examples:
        >>> CollectionTheme.GRID.value
        'grid'
        >>> CollectionTheme.LIST.value
        'list'
    """

    DEFAULT = "default"
    MINIMAL = "minimal"
    GRID = "grid"
    LIST = "list"


VALID_THEMES = frozenset(t.value for t in CollectionTheme)

# Sort options for collection items
SortOption = Literal["position", "created", "updated", "likes", "downloads"]
VALID_SORT_OPTIONS = frozenset({"position", "created", "updated", "likes", "downloads"})


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Configuration for a collection.

    Attributes:
        title: Collection title.
        description: Collection description.
        private: Whether the collection is private.
        theme: Display theme.
        namespace: Owner namespace.

    Examples:
        >>> config = CollectionConfig(
        ...     title="LLM Collection",
        ...     description="My favorite LLMs",
        ...     private=False,
        ...     theme=CollectionTheme.GRID,
        ...     namespace="username",
        ... )
        >>> config.title
        'LLM Collection'
    """

    title: str
    description: str
    private: bool
    theme: CollectionTheme
    namespace: str


@dataclass(frozen=True, slots=True)
class CollectionItem:
    """Represents an item in a collection.

    Attributes:
        item_id: Repository ID (e.g., "username/model-name").
        item_type: Type of item (model, dataset, space, paper).
        note: Optional note about the item.
        position: Position in collection.

    Examples:
        >>> item = CollectionItem(
        ...     item_id="meta-llama/Llama-2-7b",
        ...     item_type=CollectionItemType.MODEL,
        ...     note="Great base model",
        ...     position=0,
        ... )
        >>> item.item_id
        'meta-llama/Llama-2-7b'
    """

    item_id: str
    item_type: CollectionItemType
    note: str
    position: int


@dataclass(frozen=True, slots=True)
class CollectionMetadata:
    """Metadata about a collection.

    Attributes:
        slug: Collection slug/URL identifier.
        item_count: Number of items in collection.
        likes: Number of likes.
        is_featured: Whether collection is featured.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.

    Examples:
        >>> meta = CollectionMetadata(
        ...     slug="username/my-collection-abc123",
        ...     item_count=10,
        ...     likes=50,
        ...     is_featured=False,
        ...     created_at="2024-01-01T00:00:00Z",
        ...     updated_at="2024-06-01T00:00:00Z",
        ... )
        >>> meta.item_count
        10
    """

    slug: str
    item_count: int
    likes: int
    is_featured: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CollectionQuery:
    """Query parameters for searching collections.

    Attributes:
        owner: Filter by owner username.
        item: Filter by item ID.
        sort: Sort order.
        limit: Maximum results.

    Examples:
        >>> query = CollectionQuery(
        ...     owner="huggingface",
        ...     item=None,
        ...     sort="likes",
        ...     limit=10,
        ... )
        >>> query.sort
        'likes'
    """

    owner: str | None
    item: str | None
    sort: SortOption
    limit: int


@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Statistics for a collection.

    Attributes:
        total_models: Number of models.
        total_datasets: Number of datasets.
        total_spaces: Number of spaces.
        total_papers: Number of papers.
        total_likes: Total likes across items.
        total_downloads: Total downloads across items.

    Examples:
        >>> stats = CollectionStats(
        ...     total_models=5,
        ...     total_datasets=3,
        ...     total_spaces=2,
        ...     total_papers=1,
        ...     total_likes=1000,
        ...     total_downloads=50000,
        ... )
        >>> stats.total_models
        5
    """

    total_models: int
    total_datasets: int
    total_spaces: int
    total_papers: int
    total_likes: int
    total_downloads: int


def validate_collection_config(config: CollectionConfig) -> None:
    """Validate collection configuration.

    Args:
        config: Collection configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = CollectionConfig(
        ...     "My Collection", "Description", False, CollectionTheme.DEFAULT, "user"
        ... )
        >>> validate_collection_config(config)  # No error

        >>> bad = CollectionConfig(
        ...     "", "Description", False, CollectionTheme.DEFAULT, "user"
        ... )
        >>> validate_collection_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: title cannot be empty
    """
    if not config.title:
        msg = "title cannot be empty"
        raise ValueError(msg)

    if not config.namespace:
        msg = "namespace cannot be empty"
        raise ValueError(msg)

    if len(config.title) > 100:
        msg = f"title cannot exceed 100 characters, got {len(config.title)}"
        raise ValueError(msg)


def validate_collection_item(item: CollectionItem) -> None:
    """Validate collection item.

    Args:
        item: Collection item to validate.

    Raises:
        ValueError: If item is invalid.

    Examples:
        >>> item = CollectionItem("user/repo", CollectionItemType.MODEL, "", 0)
        >>> validate_collection_item(item)  # No error

        >>> bad = CollectionItem("", CollectionItemType.MODEL, "", 0)
        >>> validate_collection_item(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: item_id cannot be empty
    """
    if not item.item_id:
        msg = "item_id cannot be empty"
        raise ValueError(msg)

    if "/" not in item.item_id:
        msg = f"item_id must be in 'owner/name' format, got '{item.item_id}'"
        raise ValueError(msg)

    if item.position < 0:
        msg = f"position must be non-negative, got {item.position}"
        raise ValueError(msg)


def create_collection_config(
    title: str,
    description: str = "",
    private: bool = False,
    theme: str = "default",
    namespace: str = "",
) -> CollectionConfig:
    """Create a collection configuration.

    Args:
        title: Collection title.
        description: Collection description. Defaults to "".
        private: Whether private. Defaults to False.
        theme: Display theme. Defaults to "default".
        namespace: Owner namespace. Defaults to "".

    Returns:
        CollectionConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_collection_config("My LLMs", namespace="username")
        >>> config.title
        'My LLMs'

        >>> create_collection_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: title cannot be empty
    """
    if theme not in VALID_THEMES:
        msg = f"theme must be one of {VALID_THEMES}, got '{theme}'"
        raise ValueError(msg)

    config = CollectionConfig(
        title=title,
        description=description,
        private=private,
        theme=CollectionTheme(theme),
        namespace=namespace if namespace else "default",
    )
    validate_collection_config(config)
    return config


def create_collection_item(
    item_id: str,
    item_type: str = "model",
    note: str = "",
    position: int = 0,
) -> CollectionItem:
    """Create a collection item.

    Args:
        item_id: Repository ID (e.g., "username/repo").
        item_type: Type of item. Defaults to "model".
        note: Optional note. Defaults to "".
        position: Position in collection. Defaults to 0.

    Returns:
        CollectionItem with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> item = create_collection_item("meta-llama/Llama-2-7b")
        >>> item.item_id
        'meta-llama/Llama-2-7b'

        >>> item = create_collection_item("user/dataset", item_type="dataset")
        >>> item.item_type
        <CollectionItemType.DATASET: 'dataset'>

        >>> create_collection_item("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: item_id cannot be empty
    """
    if item_type not in VALID_ITEM_TYPES:
        msg = f"item_type must be one of {VALID_ITEM_TYPES}, got '{item_type}'"
        raise ValueError(msg)

    item = CollectionItem(
        item_id=item_id,
        item_type=CollectionItemType(item_type),
        note=note,
        position=position,
    )
    validate_collection_item(item)
    return item


def create_collection_query(
    owner: str | None = None,
    item: str | None = None,
    sort: SortOption = "likes",
    limit: int = 20,
) -> CollectionQuery:
    """Create a collection query.

    Args:
        owner: Filter by owner. Defaults to None.
        item: Filter by item ID. Defaults to None.
        sort: Sort order. Defaults to "likes".
        limit: Maximum results. Defaults to 20.

    Returns:
        CollectionQuery with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> query = create_collection_query(owner="huggingface", limit=10)
        >>> query.owner
        'huggingface'

        >>> create_collection_query(limit=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: limit must be positive
    """
    if sort not in VALID_SORT_OPTIONS:
        msg = f"sort must be one of {VALID_SORT_OPTIONS}, got '{sort}'"
        raise ValueError(msg)

    if limit <= 0:
        msg = f"limit must be positive, got {limit}"
        raise ValueError(msg)

    if limit > 100:
        msg = f"limit cannot exceed 100, got {limit}"
        raise ValueError(msg)

    return CollectionQuery(
        owner=owner,
        item=item,
        sort=sort,
        limit=limit,
    )


def create_collection_stats(
    total_models: int = 0,
    total_datasets: int = 0,
    total_spaces: int = 0,
    total_papers: int = 0,
    total_likes: int = 0,
    total_downloads: int = 0,
) -> CollectionStats:
    """Create collection statistics.

    Args:
        total_models: Number of models. Defaults to 0.
        total_datasets: Number of datasets. Defaults to 0.
        total_spaces: Number of spaces. Defaults to 0.
        total_papers: Number of papers. Defaults to 0.
        total_likes: Total likes. Defaults to 0.
        total_downloads: Total downloads. Defaults to 0.

    Returns:
        CollectionStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_collection_stats(total_models=5, total_likes=100)
        >>> stats.total_models
        5

        >>> create_collection_stats(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     total_models=-1
        ... )
        Traceback (most recent call last):
        ValueError: total_models must be non-negative
    """
    if total_models < 0:
        msg = f"total_models must be non-negative, got {total_models}"
        raise ValueError(msg)

    if total_datasets < 0:
        msg = f"total_datasets must be non-negative, got {total_datasets}"
        raise ValueError(msg)

    if total_spaces < 0:
        msg = f"total_spaces must be non-negative, got {total_spaces}"
        raise ValueError(msg)

    if total_papers < 0:
        msg = f"total_papers must be non-negative, got {total_papers}"
        raise ValueError(msg)

    if total_likes < 0:
        msg = f"total_likes must be non-negative, got {total_likes}"
        raise ValueError(msg)

    if total_downloads < 0:
        msg = f"total_downloads must be non-negative, got {total_downloads}"
        raise ValueError(msg)

    return CollectionStats(
        total_models=total_models,
        total_datasets=total_datasets,
        total_spaces=total_spaces,
        total_papers=total_papers,
        total_likes=total_likes,
        total_downloads=total_downloads,
    )


def list_item_types() -> list[str]:
    """List supported collection item types.

    Returns:
        Sorted list of item type names.

    Examples:
        >>> types = list_item_types()
        >>> "model" in types
        True
        >>> "dataset" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ITEM_TYPES)


def list_themes() -> list[str]:
    """List supported collection themes.

    Returns:
        Sorted list of theme names.

    Examples:
        >>> themes = list_themes()
        >>> "default" in themes
        True
        >>> "grid" in themes
        True
        >>> themes == sorted(themes)
        True
    """
    return sorted(VALID_THEMES)


def list_sort_options() -> list[str]:
    """List supported sort options.

    Returns:
        Sorted list of sort option names.

    Examples:
        >>> options = list_sort_options()
        >>> "likes" in options
        True
        >>> "downloads" in options
        True
        >>> options == sorted(options)
        True
    """
    return sorted(VALID_SORT_OPTIONS)


def get_item_type(name: str) -> CollectionItemType:
    """Get collection item type from name.

    Args:
        name: Item type name.

    Returns:
        CollectionItemType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_item_type("model")
        <CollectionItemType.MODEL: 'model'>

        >>> get_item_type("dataset")
        <CollectionItemType.DATASET: 'dataset'>

        >>> get_item_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: item type must be one of
    """
    if name not in VALID_ITEM_TYPES:
        msg = f"item type must be one of {VALID_ITEM_TYPES}, got '{name}'"
        raise ValueError(msg)
    return CollectionItemType(name)


def format_collection_slug(namespace: str, title: str, unique_id: str) -> str:
    """Format a collection slug.

    Args:
        namespace: Owner namespace.
        title: Collection title.
        unique_id: Unique identifier.

    Returns:
        Formatted collection slug.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> format_collection_slug("username", "My Collection", "abc123")
        'username/my-collection-abc123'

        >>> format_collection_slug(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "", "Title", "id"
        ... )
        Traceback (most recent call last):
        ValueError: namespace cannot be empty
    """
    if not namespace:
        msg = "namespace cannot be empty"
        raise ValueError(msg)

    if not title:
        msg = "title cannot be empty"
        raise ValueError(msg)

    if not unique_id:
        msg = "unique_id cannot be empty"
        raise ValueError(msg)

    # Convert title to slug format
    slug_title = title.lower().replace(" ", "-")
    # Remove non-alphanumeric characters except hyphens
    slug_title = "".join(c for c in slug_title if c.isalnum() or c == "-")

    return f"{namespace}/{slug_title}-{unique_id}"


def calculate_collection_score(stats: CollectionStats) -> float:
    """Calculate a relevance score for a collection.

    Args:
        stats: Collection statistics.

    Returns:
        Relevance score (higher is better).

    Examples:
        >>> stats = create_collection_stats(total_models=10, total_likes=100)
        >>> score = calculate_collection_score(stats)
        >>> score > 0
        True
    """
    # Weight different factors
    item_score = (
        stats.total_models * 2.0
        + stats.total_datasets * 1.5
        + stats.total_spaces * 1.0
        + stats.total_papers * 0.5
    )

    engagement_score = stats.total_likes * 0.1 + stats.total_downloads * 0.001

    return item_score + engagement_score
