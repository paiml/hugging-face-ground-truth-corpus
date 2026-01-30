"""Leaderboard integration utilities for model evaluation.

This module provides utilities for interacting with HuggingFace
leaderboards, submitting results, and tracking model rankings.

Examples:
    >>> from hf_gtc.evaluation.leaderboards import LeaderboardConfig
    >>> config = LeaderboardConfig(name="open-llm-leaderboard")
    >>> config.name
    'open-llm-leaderboard'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class LeaderboardCategory(Enum):
    """Categories of leaderboards.

    Attributes:
        LLM: Large language model leaderboards.
        VISION: Computer vision leaderboards.
        SPEECH: Speech processing leaderboards.
        MULTIMODAL: Multimodal model leaderboards.
        CUSTOM: Custom user-defined leaderboards.

    Examples:
        >>> LeaderboardCategory.LLM.value
        'llm'
        >>> LeaderboardCategory.VISION.value
        'vision'
    """

    LLM = "llm"
    VISION = "vision"
    SPEECH = "speech"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


VALID_CATEGORIES = frozenset(c.value for c in LeaderboardCategory)


class SubmissionStatus(Enum):
    """Status of a leaderboard submission.

    Attributes:
        PENDING: Submission is pending review.
        RUNNING: Evaluation is running.
        COMPLETED: Evaluation completed successfully.
        FAILED: Evaluation failed.
        REJECTED: Submission was rejected.

    Examples:
        >>> SubmissionStatus.PENDING.value
        'pending'
        >>> SubmissionStatus.COMPLETED.value
        'completed'
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


VALID_STATUSES = frozenset(s.value for s in SubmissionStatus)


@dataclass(frozen=True, slots=True)
class LeaderboardConfig:
    """Configuration for leaderboard interaction.

    Attributes:
        name: Name of the leaderboard.
        category: Leaderboard category. Defaults to LLM.
        url: URL of the leaderboard. Defaults to None.
        api_endpoint: API endpoint for submissions. Defaults to None.

    Examples:
        >>> config = LeaderboardConfig(name="open-llm-leaderboard")
        >>> config.name
        'open-llm-leaderboard'
        >>> config.category
        <LeaderboardCategory.LLM: 'llm'>
    """

    name: str
    category: LeaderboardCategory = LeaderboardCategory.LLM
    url: str | None = None
    api_endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class ModelScore:
    """Score for a model on a specific metric.

    Attributes:
        metric_name: Name of the metric.
        score: Numeric score value.
        is_higher_better: Whether higher scores are better. Defaults to True.

    Examples:
        >>> score = ModelScore("accuracy", 0.95)
        >>> score.metric_name
        'accuracy'
        >>> score.score
        0.95
    """

    metric_name: str
    score: float
    is_higher_better: bool = True


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """Entry on a leaderboard.

    Attributes:
        model_name: Name of the model.
        rank: Current rank on the leaderboard.
        scores: List of metric scores.
        submission_date: Date of submission.
        model_size: Size of the model in parameters. Defaults to None.

    Examples:
        >>> entry = LeaderboardEntry(
        ...     model_name="gpt-4",
        ...     rank=1,
        ...     scores=[ModelScore("accuracy", 0.95)],
        ...     submission_date="2024-01-15",
        ... )
        >>> entry.rank
        1
    """

    model_name: str
    rank: int
    scores: list[ModelScore]
    submission_date: str
    model_size: int | None = None


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Result of a leaderboard submission.

    Attributes:
        submission_id: Unique submission identifier.
        status: Current submission status.
        scores: Evaluation scores (if completed).
        rank: Achieved rank (if completed).
        error_message: Error message (if failed).

    Examples:
        >>> result = SubmissionResult(
        ...     submission_id="sub-001",
        ...     status=SubmissionStatus.COMPLETED,
        ...     scores=[ModelScore("accuracy", 0.92)],
        ...     rank=5,
        ...     error_message=None,
        ... )
        >>> result.status
        <SubmissionStatus.COMPLETED: 'completed'>
    """

    submission_id: str
    status: SubmissionStatus
    scores: list[ModelScore] | None
    rank: int | None
    error_message: str | None


@dataclass
class Leaderboard:
    """Leaderboard instance for tracking model rankings.

    Attributes:
        config: Leaderboard configuration.
        entries: List of leaderboard entries.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> board = Leaderboard(config)
        >>> board.config.name
        'test'
    """

    config: LeaderboardConfig
    entries: list[LeaderboardEntry] = field(default_factory=list)


def validate_leaderboard_config(config: LeaderboardConfig) -> None:
    """Validate leaderboard configuration.

    Args:
        config: LeaderboardConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If name is empty.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> validate_leaderboard_config(config)  # No error

        >>> validate_leaderboard_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = LeaderboardConfig(name="")
        >>> validate_leaderboard_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.name:
        msg = "name cannot be empty"
        raise ValueError(msg)


def create_leaderboard(config: LeaderboardConfig) -> Leaderboard:
    """Create a leaderboard with the given configuration.

    Args:
        config: Leaderboard configuration.

    Returns:
        Configured Leaderboard instance.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> board = create_leaderboard(config)
        >>> board.config.name
        'test'

        >>> create_leaderboard(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_leaderboard_config(config)
    return Leaderboard(config=config)


def add_entry(
    leaderboard: Leaderboard,
    entry: LeaderboardEntry,
) -> None:
    """Add an entry to the leaderboard.

    Args:
        leaderboard: Leaderboard to add entry to.
        entry: Entry to add.

    Raises:
        ValueError: If leaderboard is None.
        ValueError: If entry is None.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> board = create_leaderboard(config)
        >>> entry = LeaderboardEntry("model", 1, [], "2024-01-01")
        >>> add_entry(board, entry)
        >>> len(board.entries)
        1

        >>> add_entry(None, entry)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: leaderboard cannot be None
    """
    if leaderboard is None:
        msg = "leaderboard cannot be None"
        raise ValueError(msg)

    if entry is None:
        msg = "entry cannot be None"
        raise ValueError(msg)

    leaderboard.entries.append(entry)


def get_top_entries(
    leaderboard: Leaderboard,
    n: int = 10,
) -> list[LeaderboardEntry]:
    """Get top N entries from the leaderboard.

    Args:
        leaderboard: Leaderboard to query.
        n: Number of entries to return. Defaults to 10.

    Returns:
        List of top N entries sorted by rank.

    Raises:
        ValueError: If leaderboard is None.
        ValueError: If n is not positive.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> board = create_leaderboard(config)
        >>> entry1 = LeaderboardEntry("m1", 2, [], "2024-01-01")
        >>> entry2 = LeaderboardEntry("m2", 1, [], "2024-01-01")
        >>> add_entry(board, entry1)
        >>> add_entry(board, entry2)
        >>> top = get_top_entries(board, 2)
        >>> top[0].model_name
        'm2'

        >>> get_top_entries(None, 5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: leaderboard cannot be None
    """
    if leaderboard is None:
        msg = "leaderboard cannot be None"
        raise ValueError(msg)

    if n <= 0:
        msg = f"n must be positive, got {n}"
        raise ValueError(msg)

    sorted_entries = sorted(leaderboard.entries, key=lambda e: e.rank)
    return sorted_entries[:n]


def find_entry_by_model(
    leaderboard: Leaderboard,
    model_name: str,
) -> LeaderboardEntry | None:
    """Find a leaderboard entry by model name.

    Args:
        leaderboard: Leaderboard to search.
        model_name: Name of the model to find.

    Returns:
        LeaderboardEntry if found, None otherwise.

    Raises:
        ValueError: If leaderboard is None.
        ValueError: If model_name is empty.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> board = create_leaderboard(config)
        >>> entry = LeaderboardEntry("my-model", 1, [], "2024-01-01")
        >>> add_entry(board, entry)
        >>> found = find_entry_by_model(board, "my-model")
        >>> found.rank
        1
        >>> find_entry_by_model(board, "unknown") is None
        True

        >>> find_entry_by_model(None, "test")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: leaderboard cannot be None
    """
    if leaderboard is None:
        msg = "leaderboard cannot be None"
        raise ValueError(msg)

    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    for entry in leaderboard.entries:
        if entry.model_name == model_name:
            return entry

    return None


def compute_average_score(entry: LeaderboardEntry) -> float:
    """Compute average score across all metrics for an entry.

    Args:
        entry: Leaderboard entry.

    Returns:
        Average score value.

    Raises:
        ValueError: If entry is None.

    Examples:
        >>> scores = [ModelScore("a", 0.8), ModelScore("b", 0.9)]
        >>> entry = LeaderboardEntry("model", 1, scores, "2024-01-01")
        >>> round(compute_average_score(entry), 2)
        0.85

        >>> empty_entry = LeaderboardEntry("model", 1, [], "2024-01-01")
        >>> compute_average_score(empty_entry)
        0.0

        >>> compute_average_score(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: entry cannot be None
    """
    if entry is None:
        msg = "entry cannot be None"
        raise ValueError(msg)

    if not entry.scores:
        return 0.0

    total = sum(s.score for s in entry.scores)
    return total / len(entry.scores)


def get_score_by_metric(
    entry: LeaderboardEntry,
    metric_name: str,
) -> float | None:
    """Get score for a specific metric from an entry.

    Args:
        entry: Leaderboard entry.
        metric_name: Name of the metric.

    Returns:
        Score value if found, None otherwise.

    Raises:
        ValueError: If entry is None.
        ValueError: If metric_name is empty.

    Examples:
        >>> scores = [ModelScore("accuracy", 0.95), ModelScore("f1", 0.92)]
        >>> entry = LeaderboardEntry("model", 1, scores, "2024-01-01")
        >>> get_score_by_metric(entry, "accuracy")
        0.95
        >>> get_score_by_metric(entry, "unknown") is None
        True

        >>> get_score_by_metric(None, "test")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: entry cannot be None
    """
    if entry is None:
        msg = "entry cannot be None"
        raise ValueError(msg)

    if not metric_name:
        msg = "metric_name cannot be empty"
        raise ValueError(msg)

    for score in entry.scores:
        if score.metric_name == metric_name:
            return score.score

    return None


def create_submission(
    model_name: str,
    model_path: str,
    *,
    model_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a submission payload for a leaderboard.

    Args:
        model_name: Name of the model.
        model_path: Path or identifier of the model.
        model_size: Size of the model in parameters. Defaults to None.
        metadata: Additional metadata. Defaults to None.

    Returns:
        Dictionary with submission payload.

    Raises:
        ValueError: If model_name is empty.
        ValueError: If model_path is empty.

    Examples:
        >>> payload = create_submission("my-model", "org/my-model")
        >>> payload["model_name"]
        'my-model'
        >>> "submission_date" in payload
        True

        >>> create_submission("", "path")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if not model_path:
        msg = "model_path cannot be empty"
        raise ValueError(msg)

    payload = {
        "model_name": model_name,
        "model_path": model_path,
        "submission_date": datetime.now().isoformat(),
    }

    if model_size is not None:
        payload["model_size"] = model_size

    if metadata:
        payload["metadata"] = metadata

    return payload


def parse_submission_result(
    response: dict[str, Any],
) -> SubmissionResult:
    """Parse a submission result from API response.

    Args:
        response: API response dictionary.

    Returns:
        Parsed SubmissionResult.

    Raises:
        ValueError: If response is None.
        ValueError: If required fields are missing.

    Examples:
        >>> response = {
        ...     "submission_id": "sub-001",
        ...     "status": "completed",
        ...     "scores": [{"metric_name": "acc", "score": 0.9}],
        ...     "rank": 5,
        ... }
        >>> result = parse_submission_result(response)
        >>> result.status
        <SubmissionStatus.COMPLETED: 'completed'>

        >>> parse_submission_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: response cannot be None
    """
    if response is None:
        msg = "response cannot be None"
        raise ValueError(msg)

    if "submission_id" not in response:
        msg = "response missing required field: submission_id"
        raise ValueError(msg)

    if "status" not in response:
        msg = "response missing required field: status"
        raise ValueError(msg)

    status = SubmissionStatus(response["status"])

    scores = None
    raw_scores = response.get("scores")
    if raw_scores:
        scores = [
            ModelScore(
                metric_name=s["metric_name"],
                score=s["score"],
                is_higher_better=s.get("is_higher_better", True),
            )
            for s in raw_scores
        ]

    return SubmissionResult(
        submission_id=response["submission_id"],
        status=status,
        scores=scores,
        rank=response.get("rank"),
        error_message=response.get("error_message"),
    )


def compare_entries(
    entry1: LeaderboardEntry,
    entry2: LeaderboardEntry,
) -> dict[str, Any]:
    """Compare two leaderboard entries.

    Args:
        entry1: First entry to compare.
        entry2: Second entry to compare.

    Returns:
        Dictionary with comparison results.

    Raises:
        ValueError: If entry1 is None.
        ValueError: If entry2 is None.

    Examples:
        >>> e1 = LeaderboardEntry("m1", 1, [ModelScore("acc", 0.9)], "2024-01-01")
        >>> e2 = LeaderboardEntry("m2", 2, [ModelScore("acc", 0.85)], "2024-01-02")
        >>> cmp = compare_entries(e1, e2)
        >>> cmp["rank_difference"]
        -1
        >>> cmp["better_ranked"]
        'm1'

        >>> compare_entries(None, e2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: entry1 cannot be None
    """
    if entry1 is None:
        msg = "entry1 cannot be None"
        raise ValueError(msg)

    if entry2 is None:
        msg = "entry2 cannot be None"
        raise ValueError(msg)

    avg1 = compute_average_score(entry1)
    avg2 = compute_average_score(entry2)

    better = entry1.model_name if entry1.rank < entry2.rank else entry2.model_name
    higher = entry1.model_name if avg1 > avg2 else entry2.model_name
    return {
        "model1": entry1.model_name,
        "model2": entry2.model_name,
        "rank_difference": entry1.rank - entry2.rank,
        "score_difference": avg1 - avg2,
        "better_ranked": better,
        "higher_score": higher,
    }


def filter_entries_by_size(
    entries: Sequence[LeaderboardEntry],
    min_size: int | None = None,
    max_size: int | None = None,
) -> list[LeaderboardEntry]:
    """Filter entries by model size.

    Args:
        entries: Entries to filter.
        min_size: Minimum model size (inclusive). Defaults to None.
        max_size: Maximum model size (inclusive). Defaults to None.

    Returns:
        Filtered list of entries.

    Raises:
        ValueError: If entries is None.

    Examples:
        >>> e1 = LeaderboardEntry("m1", 1, [], "2024-01-01", model_size=7000000000)
        >>> e2 = LeaderboardEntry("m2", 2, [], "2024-01-01", model_size=13000000000)
        >>> filtered = filter_entries_by_size([e1, e2], max_size=10000000000)
        >>> len(filtered)
        1
        >>> filtered[0].model_name
        'm1'

        >>> filter_entries_by_size(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: entries cannot be None
    """
    if entries is None:
        msg = "entries cannot be None"
        raise ValueError(msg)

    result = []
    for entry in entries:
        if entry.model_size is None:
            continue

        if min_size is not None and entry.model_size < min_size:
            continue

        if max_size is not None and entry.model_size > max_size:
            continue

        result.append(entry)

    return result


def list_categories() -> list[str]:
    """List all available leaderboard categories.

    Returns:
        Sorted list of category names.

    Examples:
        >>> categories = list_categories()
        >>> "llm" in categories
        True
        >>> "vision" in categories
        True
        >>> categories == sorted(categories)
        True
    """
    return sorted(VALID_CATEGORIES)


def validate_category(category: str) -> bool:
    """Validate if a string is a valid leaderboard category.

    Args:
        category: The category string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_category("llm")
        True
        >>> validate_category("vision")
        True
        >>> validate_category("invalid")
        False
        >>> validate_category("")
        False
    """
    return category in VALID_CATEGORIES


def get_category(name: str) -> LeaderboardCategory:
    """Get LeaderboardCategory enum from string name.

    Args:
        name: Name of the category.

    Returns:
        Corresponding LeaderboardCategory enum value.

    Raises:
        ValueError: If name is not a valid category.

    Examples:
        >>> get_category("llm")
        <LeaderboardCategory.LLM: 'llm'>

        >>> get_category("vision")
        <LeaderboardCategory.VISION: 'vision'>

        >>> get_category("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid category: invalid
    """
    if not validate_category(name):
        msg = f"invalid category: {name}"
        raise ValueError(msg)

    return LeaderboardCategory(name)


def list_submission_statuses() -> list[str]:
    """List all available submission statuses.

    Returns:
        Sorted list of status names.

    Examples:
        >>> statuses = list_submission_statuses()
        >>> "completed" in statuses
        True
        >>> "pending" in statuses
        True
        >>> statuses == sorted(statuses)
        True
    """
    return sorted(VALID_STATUSES)


def validate_submission_status(status: str) -> bool:
    """Validate if a string is a valid submission status.

    Args:
        status: The status string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_submission_status("completed")
        True
        >>> validate_submission_status("pending")
        True
        >>> validate_submission_status("invalid")
        False
        >>> validate_submission_status("")
        False
    """
    return status in VALID_STATUSES


def format_leaderboard(
    leaderboard: Leaderboard,
    max_entries: int = 10,
) -> str:
    """Format leaderboard as a human-readable string.

    Args:
        leaderboard: Leaderboard to format.
        max_entries: Maximum entries to show. Defaults to 10.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If leaderboard is None.

    Examples:
        >>> config = LeaderboardConfig(name="Test Board")
        >>> board = create_leaderboard(config)
        >>> "Test Board" in format_leaderboard(board)
        True

        >>> format_leaderboard(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: leaderboard cannot be None
    """
    if leaderboard is None:
        msg = "leaderboard cannot be None"
        raise ValueError(msg)

    lines = [
        f"Leaderboard: {leaderboard.config.name}",
        f"Category: {leaderboard.config.category.value}",
        f"Total Entries: {len(leaderboard.entries)}",
        "",
    ]

    if leaderboard.entries:
        lines.append("Top Entries:")
        top_entries = get_top_entries(leaderboard, max_entries)
        for entry in top_entries:
            avg_score = compute_average_score(entry)
            lines.append(f"  {entry.rank}. {entry.model_name} (avg: {avg_score:.4f})")
    else:
        lines.append("No entries yet.")

    return "\n".join(lines)


def compute_leaderboard_stats(leaderboard: Leaderboard) -> dict[str, Any]:
    """Compute statistics for a leaderboard.

    Args:
        leaderboard: Leaderboard to analyze.

    Returns:
        Dictionary with leaderboard statistics.

    Raises:
        ValueError: If leaderboard is None.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> board = create_leaderboard(config)
        >>> stats = compute_leaderboard_stats(board)
        >>> stats["total_entries"]
        0

        >>> compute_leaderboard_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: leaderboard cannot be None
    """
    if leaderboard is None:
        msg = "leaderboard cannot be None"
        raise ValueError(msg)

    if not leaderboard.entries:
        return {
            "total_entries": 0,
            "avg_score": 0.0,
            "score_range": (0.0, 0.0),
            "models_with_size": 0,
        }

    avg_scores = [compute_average_score(e) for e in leaderboard.entries]
    models_with_size = sum(1 for e in leaderboard.entries if e.model_size is not None)

    return {
        "total_entries": len(leaderboard.entries),
        "avg_score": sum(avg_scores) / len(avg_scores),
        "score_range": (min(avg_scores), max(avg_scores)),
        "models_with_size": models_with_size,
    }
