"""Leaderboard integration utilities for model evaluation.

This module provides utilities for interacting with HuggingFace
leaderboards, submitting results, and tracking model rankings.

Examples:
    >>> from hf_gtc.evaluation.leaderboards import LeaderboardConfig
    >>> config = LeaderboardConfig(name="open-llm-leaderboard")
    >>> config.name
    'open-llm-leaderboard'
    >>> from hf_gtc.evaluation.leaderboards import LeaderboardType, RankingMethod
    >>> LeaderboardType.OPEN_LLM.value
    'open_llm'
    >>> RankingMethod.ELO.value
    'elo'
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class LeaderboardType(Enum):
    """Types of HuggingFace leaderboards.

    Attributes:
        OPEN_LLM: Open LLM Leaderboard for general LLM evaluation.
        MTEB: Massive Text Embedding Benchmark leaderboard.
        HELM: Holistic Evaluation of Language Models leaderboard.
        BIGCODE: BigCode leaderboard for code generation models.
        CHATBOT_ARENA: Chatbot Arena with ELO-based rankings.

    Examples:
        >>> LeaderboardType.OPEN_LLM.value
        'open_llm'
        >>> LeaderboardType.MTEB.value
        'mteb'
        >>> LeaderboardType.CHATBOT_ARENA.value
        'chatbot_arena'
    """

    OPEN_LLM = "open_llm"
    MTEB = "mteb"
    HELM = "helm"
    BIGCODE = "bigcode"
    CHATBOT_ARENA = "chatbot_arena"


VALID_LEADERBOARD_TYPES = frozenset(t.value for t in LeaderboardType)


class RankingMethod(Enum):
    """Methods for ranking models on leaderboards.

    Attributes:
        SCORE: Simple score-based ranking (higher is better).
        ELO: ELO rating system (used in Chatbot Arena).
        WIN_RATE: Win rate percentage ranking.
        WEIGHTED: Weighted combination of multiple metrics.

    Examples:
        >>> RankingMethod.SCORE.value
        'score'
        >>> RankingMethod.ELO.value
        'elo'
        >>> RankingMethod.WIN_RATE.value
        'win_rate'
    """

    SCORE = "score"
    ELO = "elo"
    WIN_RATE = "win_rate"
    WEIGHTED = "weighted"


VALID_RANKING_METHODS = frozenset(m.value for m in RankingMethod)


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
        leaderboard_type: Type of leaderboard. Defaults to OPEN_LLM.
        metrics: Tuple of metric names to track.
        ranking_method: Method for ranking models. Defaults to SCORE.
        higher_is_better: Whether higher scores are better. Defaults to True.
        url: URL of the leaderboard. Defaults to None.
        api_endpoint: API endpoint for submissions. Defaults to None.

    Examples:
        >>> config = LeaderboardConfig(name="open-llm-leaderboard")
        >>> config.name
        'open-llm-leaderboard'
        >>> config.category
        <LeaderboardCategory.LLM: 'llm'>
        >>> config.leaderboard_type
        <LeaderboardType.OPEN_LLM: 'open_llm'>
        >>> config.higher_is_better
        True
    """

    name: str
    category: LeaderboardCategory = LeaderboardCategory.LLM
    leaderboard_type: LeaderboardType = LeaderboardType.OPEN_LLM
    metrics: tuple[str, ...] = ()
    ranking_method: RankingMethod = RankingMethod.SCORE
    higher_is_better: bool = True
    url: str | None = None
    api_endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    """Configuration for leaderboard submission.

    Attributes:
        model_name: Name of the model to submit.
        revision: Model revision (commit hash or branch). Defaults to "main".
        precision: Model precision (e.g., "float16", "bfloat16"). Defaults to "float16".
        num_few_shot: Number of few-shot examples. Defaults to 0.

    Examples:
        >>> config = SubmissionConfig(model_name="meta-llama/Llama-2-7b")
        >>> config.model_name
        'meta-llama/Llama-2-7b'
        >>> config.revision
        'main'
        >>> config.precision
        'float16'
        >>> config.num_few_shot
        0

        >>> config = SubmissionConfig(
        ...     model_name="mistralai/Mistral-7B-v0.1",
        ...     revision="abc123",
        ...     precision="bfloat16",
        ...     num_few_shot=5,
        ... )
        >>> config.num_few_shot
        5
    """

    model_name: str
    revision: str = "main"
    precision: str = "float16"
    num_few_shot: int = 0


@dataclass(frozen=True, slots=True)
class LeaderboardStats:
    """Statistics summary for a leaderboard.

    Attributes:
        total_models: Total number of models on the leaderboard.
        avg_score: Average score across all models.
        top_score: Highest score on the leaderboard.
        last_updated: ISO format timestamp of last update.

    Examples:
        >>> stats = LeaderboardStats(
        ...     total_models=100,
        ...     avg_score=0.65,
        ...     top_score=0.89,
        ...     last_updated="2024-01-15T12:00:00",
        ... )
        >>> stats.total_models
        100
        >>> stats.avg_score
        0.65
        >>> stats.top_score
        0.89
    """

    total_models: int
    avg_score: float
    top_score: float
    last_updated: str


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


# New enum list/get/validate functions for LeaderboardType


def list_leaderboard_types() -> list[str]:
    """List all available leaderboard types.

    Returns:
        Sorted list of leaderboard type names.

    Examples:
        >>> types = list_leaderboard_types()
        >>> "open_llm" in types
        True
        >>> "mteb" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_LEADERBOARD_TYPES)


def get_leaderboard_type(name: str) -> LeaderboardType:
    """Get LeaderboardType enum from string name.

    Args:
        name: Name of the leaderboard type.

    Returns:
        Corresponding LeaderboardType enum value.

    Raises:
        ValueError: If name is not a valid leaderboard type.

    Examples:
        >>> get_leaderboard_type("open_llm")
        <LeaderboardType.OPEN_LLM: 'open_llm'>

        >>> get_leaderboard_type("chatbot_arena")
        <LeaderboardType.CHATBOT_ARENA: 'chatbot_arena'>

        >>> get_leaderboard_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid leaderboard type: invalid
    """
    if name not in VALID_LEADERBOARD_TYPES:
        msg = f"invalid leaderboard type: {name}"
        raise ValueError(msg)

    return LeaderboardType(name)


def validate_leaderboard_type(name: str) -> bool:
    """Check if a leaderboard type name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_leaderboard_type("open_llm")
        True
        >>> validate_leaderboard_type("mteb")
        True
        >>> validate_leaderboard_type("invalid")
        False
        >>> validate_leaderboard_type("")
        False
    """
    return name in VALID_LEADERBOARD_TYPES


# New enum list/get/validate functions for RankingMethod


def list_ranking_methods() -> list[str]:
    """List all available ranking methods.

    Returns:
        Sorted list of ranking method names.

    Examples:
        >>> methods = list_ranking_methods()
        >>> "score" in methods
        True
        >>> "elo" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_RANKING_METHODS)


def get_ranking_method(name: str) -> RankingMethod:
    """Get RankingMethod enum from string name.

    Args:
        name: Name of the ranking method.

    Returns:
        Corresponding RankingMethod enum value.

    Raises:
        ValueError: If name is not a valid ranking method.

    Examples:
        >>> get_ranking_method("score")
        <RankingMethod.SCORE: 'score'>

        >>> get_ranking_method("elo")
        <RankingMethod.ELO: 'elo'>

        >>> get_ranking_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid ranking method: invalid
    """
    if name not in VALID_RANKING_METHODS:
        msg = f"invalid ranking method: {name}"
        raise ValueError(msg)

    return RankingMethod(name)


def validate_ranking_method(name: str) -> bool:
    """Check if a ranking method name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_ranking_method("score")
        True
        >>> validate_ranking_method("elo")
        True
        >>> validate_ranking_method("invalid")
        False
        >>> validate_ranking_method("")
        False
    """
    return name in VALID_RANKING_METHODS


# Factory functions


def create_leaderboard_config(
    name: str,
    *,
    category: LeaderboardCategory = LeaderboardCategory.LLM,
    leaderboard_type: LeaderboardType = LeaderboardType.OPEN_LLM,
    metrics: tuple[str, ...] = (),
    ranking_method: RankingMethod = RankingMethod.SCORE,
    higher_is_better: bool = True,
    url: str | None = None,
    api_endpoint: str | None = None,
) -> LeaderboardConfig:
    """Create and validate a leaderboard configuration.

    Args:
        name: Name of the leaderboard.
        category: Leaderboard category. Defaults to LLM.
        leaderboard_type: Type of leaderboard. Defaults to OPEN_LLM.
        metrics: Tuple of metric names to track.
        ranking_method: Method for ranking models. Defaults to SCORE.
        higher_is_better: Whether higher scores are better. Defaults to True.
        url: URL of the leaderboard. Defaults to None.
        api_endpoint: API endpoint for submissions. Defaults to None.

    Returns:
        Validated LeaderboardConfig instance.

    Raises:
        ValueError: If name is empty.

    Examples:
        >>> config = create_leaderboard_config("my-leaderboard")
        >>> config.name
        'my-leaderboard'

        >>> config = create_leaderboard_config(
        ...     "Open LLM",
        ...     leaderboard_type=LeaderboardType.OPEN_LLM,
        ...     metrics=("mmlu", "hellaswag", "arc"),
        ...     ranking_method=RankingMethod.WEIGHTED,
        ... )
        >>> config.leaderboard_type
        <LeaderboardType.OPEN_LLM: 'open_llm'>
        >>> config.metrics
        ('mmlu', 'hellaswag', 'arc')

        >>> create_leaderboard_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    config = LeaderboardConfig(
        name=name,
        category=category,
        leaderboard_type=leaderboard_type,
        metrics=metrics,
        ranking_method=ranking_method,
        higher_is_better=higher_is_better,
        url=url,
        api_endpoint=api_endpoint,
    )
    validate_leaderboard_config(config)
    return config


def create_submission_config(
    model_name: str,
    *,
    revision: str = "main",
    precision: str = "float16",
    num_few_shot: int = 0,
) -> SubmissionConfig:
    """Create and validate a submission configuration.

    Args:
        model_name: Name of the model to submit.
        revision: Model revision. Defaults to "main".
        precision: Model precision. Defaults to "float16".
        num_few_shot: Number of few-shot examples. Defaults to 0.

    Returns:
        Validated SubmissionConfig instance.

    Raises:
        ValueError: If model_name is empty.
        ValueError: If num_few_shot is negative.

    Examples:
        >>> config = create_submission_config("meta-llama/Llama-2-7b")
        >>> config.model_name
        'meta-llama/Llama-2-7b'

        >>> config = create_submission_config(
        ...     "mistralai/Mistral-7B-v0.1",
        ...     revision="v0.1",
        ...     precision="bfloat16",
        ...     num_few_shot=5,
        ... )
        >>> config.revision
        'v0.1'
        >>> config.num_few_shot
        5

        >>> create_submission_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    config = SubmissionConfig(
        model_name=model_name,
        revision=revision,
        precision=precision,
        num_few_shot=num_few_shot,
    )
    validate_submission_config(config)
    return config


def create_leaderboard_entry(
    model_name: str,
    scores: dict[str, float],
    rank: int,
    submission_date: str,
    *,
    model_size: int | None = None,
) -> LeaderboardEntry:
    """Create and validate a leaderboard entry.

    Args:
        model_name: Name of the model.
        scores: Dictionary mapping metric names to scores.
        rank: Current rank on the leaderboard.
        submission_date: Date of submission (ISO format).
        model_size: Size of the model in parameters. Defaults to None.

    Returns:
        Validated LeaderboardEntry instance.

    Raises:
        ValueError: If model_name is empty.
        ValueError: If rank is not positive.

    Examples:
        >>> entry = create_leaderboard_entry(
        ...     "gpt-4",
        ...     {"accuracy": 0.95, "f1": 0.92},
        ...     1,
        ...     "2024-01-15",
        ... )
        >>> entry.model_name
        'gpt-4'
        >>> entry.rank
        1

        >>> entry = create_leaderboard_entry(
        ...     "llama-7b",
        ...     {"mmlu": 0.65},
        ...     5,
        ...     "2024-01-10",
        ...     model_size=7000000000,
        ... )
        >>> entry.model_size
        7000000000

        >>> create_leaderboard_entry(
        ...     "", {}, 1, "2024-01-01"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if rank <= 0:
        msg = f"rank must be positive, got {rank}"
        raise ValueError(msg)

    model_scores = [
        ModelScore(metric_name=name, score=score)
        for name, score in scores.items()
    ]

    return LeaderboardEntry(
        model_name=model_name,
        rank=rank,
        scores=model_scores,
        submission_date=submission_date,
        model_size=model_size,
    )


# Validation functions


def validate_submission_config(config: SubmissionConfig) -> None:
    """Validate submission configuration.

    Args:
        config: SubmissionConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If model_name is empty.
        ValueError: If num_few_shot is negative.

    Examples:
        >>> config = SubmissionConfig(model_name="test-model")
        >>> validate_submission_config(config)  # No error

        >>> validate_submission_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SubmissionConfig(model_name="test", num_few_shot=-1)
        >>> validate_submission_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_few_shot cannot be negative
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if config.num_few_shot < 0:
        msg = f"num_few_shot cannot be negative, got {config.num_few_shot}"
        raise ValueError(msg)


def validate_leaderboard_stats(stats: LeaderboardStats) -> None:
    """Validate leaderboard statistics.

    Args:
        stats: LeaderboardStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If total_models is negative.
        ValueError: If avg_score is out of range.
        ValueError: If top_score is out of range.

    Examples:
        >>> stats = LeaderboardStats(100, 0.65, 0.89, "2024-01-15T12:00:00")
        >>> validate_leaderboard_stats(stats)  # No error

        >>> validate_leaderboard_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = LeaderboardStats(-1, 0.5, 0.8, "2024-01-15")
        >>> validate_leaderboard_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_models cannot be negative
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.total_models < 0:
        msg = f"total_models cannot be negative, got {stats.total_models}"
        raise ValueError(msg)

    if not 0.0 <= stats.avg_score <= 1.0:
        msg = f"avg_score must be between 0 and 1, got {stats.avg_score}"
        raise ValueError(msg)

    if not 0.0 <= stats.top_score <= 1.0:
        msg = f"top_score must be between 0 and 1, got {stats.top_score}"
        raise ValueError(msg)


def validate_submission(submission: dict[str, Any]) -> bool:
    """Validate a submission payload.

    Args:
        submission: Submission dictionary to validate.

    Returns:
        True if valid, False otherwise.

    Raises:
        ValueError: If submission is None.

    Examples:
        >>> valid = {"model_name": "test", "model_path": "org/test"}
        >>> validate_submission(valid)
        True

        >>> invalid = {"model_name": ""}
        >>> validate_submission(invalid)
        False

        >>> validate_submission(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: submission cannot be None
    """
    if submission is None:
        msg = "submission cannot be None"
        raise ValueError(msg)

    # Check required fields
    if "model_name" not in submission or not submission["model_name"]:
        return False

    return "model_path" in submission and bool(submission["model_path"])


# Core functions


def calculate_ranking(
    entries: Sequence[LeaderboardEntry],
    *,
    method: RankingMethod = RankingMethod.SCORE,
    higher_is_better: bool = True,
) -> list[LeaderboardEntry]:
    """Calculate rankings for a list of entries.

    Args:
        entries: Sequence of leaderboard entries.
        method: Ranking method to use. Defaults to SCORE.
        higher_is_better: Whether higher scores are better. Defaults to True.

    Returns:
        List of entries with updated ranks.

    Raises:
        ValueError: If entries is None.

    Examples:
        >>> e1 = LeaderboardEntry("m1", 0, [ModelScore("a", 0.8)], "2024-01-01")
        >>> e2 = LeaderboardEntry("m2", 0, [ModelScore("a", 0.9)], "2024-01-01")
        >>> ranked = calculate_ranking([e1, e2])
        >>> ranked[0].model_name
        'm2'
        >>> ranked[0].rank
        1

        >>> calculate_ranking(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: entries cannot be None
    """
    if entries is None:
        msg = "entries cannot be None"
        raise ValueError(msg)

    if len(entries) == 0:
        return []

    # Calculate average scores for sorting
    entry_scores = [
        (entry, compute_average_score(entry))
        for entry in entries
    ]

    # Sort by score (descending if higher is better)
    sorted_entries = sorted(
        entry_scores,
        key=lambda x: x[1],
        reverse=higher_is_better,
    )

    # Create new entries with updated ranks
    ranked = []
    for i, (entry, _) in enumerate(sorted_entries, start=1):
        new_entry = LeaderboardEntry(
            model_name=entry.model_name,
            rank=i,
            scores=entry.scores,
            submission_date=entry.submission_date,
            model_size=entry.model_size,
        )
        ranked.append(new_entry)

    return ranked


def compute_elo_rating(
    current_rating: float,
    opponent_rating: float,
    outcome: float,
    *,
    k_factor: float = 32.0,
) -> float:
    """Compute updated ELO rating after a match.

    Args:
        current_rating: Current ELO rating.
        opponent_rating: Opponent's ELO rating.
        outcome: Match outcome (1.0 = win, 0.5 = draw, 0.0 = loss).
        k_factor: K-factor for rating adjustment. Defaults to 32.0.

    Returns:
        Updated ELO rating.

    Raises:
        ValueError: If outcome is not in [0, 1].
        ValueError: If k_factor is not positive.

    Examples:
        >>> # Win against equal opponent
        >>> new_rating = compute_elo_rating(1500.0, 1500.0, 1.0)
        >>> new_rating > 1500.0
        True

        >>> # Loss against equal opponent
        >>> new_rating = compute_elo_rating(1500.0, 1500.0, 0.0)
        >>> new_rating < 1500.0
        True

        >>> # Win against stronger opponent
        >>> new_rating = compute_elo_rating(1400.0, 1600.0, 1.0)
        >>> new_rating - 1400.0 > 16.0  # Bigger gain
        True

        >>> compute_elo_rating(1500.0, 1500.0, 1.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: outcome must be between 0 and 1
    """
    if not 0.0 <= outcome <= 1.0:
        msg = f"outcome must be between 0 and 1, got {outcome}"
        raise ValueError(msg)

    if k_factor <= 0.0:
        msg = f"k_factor must be positive, got {k_factor}"
        raise ValueError(msg)

    # Expected score based on rating difference
    expected = 1.0 / (1.0 + math.pow(10.0, (opponent_rating - current_rating) / 400.0))

    # Update rating
    return current_rating + k_factor * (outcome - expected)


def format_submission(submission_config: SubmissionConfig) -> str:
    """Format a submission configuration as a human-readable string.

    Args:
        submission_config: SubmissionConfig to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If submission_config is None.

    Examples:
        >>> config = SubmissionConfig(
        ...     model_name="meta-llama/Llama-2-7b",
        ...     revision="main",
        ...     precision="float16",
        ...     num_few_shot=5,
        ... )
        >>> formatted = format_submission(config)
        >>> "meta-llama/Llama-2-7b" in formatted
        True
        >>> "float16" in formatted
        True

        >>> format_submission(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: submission_config cannot be None
    """
    if submission_config is None:
        msg = "submission_config cannot be None"
        raise ValueError(msg)

    lines = [
        "Submission Configuration",
        "=" * 30,
        f"Model: {submission_config.model_name}",
        f"Revision: {submission_config.revision}",
        f"Precision: {submission_config.precision}",
        f"Few-shot Examples: {submission_config.num_few_shot}",
    ]

    return "\n".join(lines)


def compare_models(
    model_a: LeaderboardEntry,
    model_b: LeaderboardEntry,
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare two models on the leaderboard.

    Args:
        model_a: First model entry.
        model_b: Second model entry.
        metrics: Specific metrics to compare (optional).

    Returns:
        Dictionary with comparison results.

    Raises:
        ValueError: If model_a is None.
        ValueError: If model_b is None.

    Examples:
        >>> e1 = LeaderboardEntry("m1", 1, [ModelScore("acc", 0.9)], "2024-01-01")
        >>> e2 = LeaderboardEntry("m2", 2, [ModelScore("acc", 0.85)], "2024-01-02")
        >>> cmp = compare_models(e1, e2)
        >>> cmp["winner"]
        'm1'
        >>> cmp["rank_difference"]
        -1

        >>> compare_models(None, e2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_a cannot be None
    """
    if model_a is None:
        msg = "model_a cannot be None"
        raise ValueError(msg)

    if model_b is None:
        msg = "model_b cannot be None"
        raise ValueError(msg)

    # Get scores
    avg_a = compute_average_score(model_a)
    avg_b = compute_average_score(model_b)

    # Determine winner
    if avg_a > avg_b:
        winner = model_a.model_name
    elif avg_b > avg_a:
        winner = model_b.model_name
    else:
        winner = "tie"

    # Compare specific metrics if provided
    per_metric: dict[str, dict[str, float | str]] = {}
    if metrics:
        for metric in metrics:
            score_a = get_score_by_metric(model_a, metric)
            score_b = get_score_by_metric(model_b, metric)

            if score_a is not None and score_b is not None:
                diff = score_a - score_b
                metric_winner = (
                    model_a.model_name if diff > 0
                    else model_b.model_name if diff < 0
                    else "tie"
                )
                per_metric[metric] = {
                    "score_a": score_a,
                    "score_b": score_b,
                    "difference": diff,
                    "winner": metric_winner,
                }

    return {
        "model_a": model_a.model_name,
        "model_b": model_b.model_name,
        "avg_score_a": avg_a,
        "avg_score_b": avg_b,
        "score_difference": avg_a - avg_b,
        "rank_difference": model_a.rank - model_b.rank,
        "winner": winner,
        "per_metric": per_metric,
    }


def format_leaderboard_stats(stats: LeaderboardStats) -> str:
    """Format leaderboard statistics as a human-readable string.

    Args:
        stats: LeaderboardStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = LeaderboardStats(
        ...     total_models=100,
        ...     avg_score=0.65,
        ...     top_score=0.89,
        ...     last_updated="2024-01-15T12:00:00",
        ... )
        >>> formatted = format_leaderboard_stats(stats)
        >>> "100" in formatted
        True
        >>> "0.65" in formatted
        True

        >>> format_leaderboard_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        "Leaderboard Statistics",
        "=" * 30,
        f"Total Models: {stats.total_models}",
        f"Average Score: {stats.avg_score:.4f}",
        f"Top Score: {stats.top_score:.4f}",
        f"Last Updated: {stats.last_updated}",
    ]

    return "\n".join(lines)


def get_recommended_leaderboard_config(
    model_type: str,
    *,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Get recommended leaderboard configuration for a model type.

    Args:
        model_type: Type of model (e.g., "llm", "embedding", "code").
        task_type: Specific task type (optional).

    Returns:
        Dictionary with recommended configuration.

    Raises:
        ValueError: If model_type is None or empty.

    Examples:
        >>> config = get_recommended_leaderboard_config("llm")
        >>> "leaderboard_type" in config
        True
        >>> config["leaderboard_type"] == LeaderboardType.OPEN_LLM
        True

        >>> config = get_recommended_leaderboard_config("embedding")
        >>> config["leaderboard_type"] == LeaderboardType.MTEB
        True

        >>> config = get_recommended_leaderboard_config("code")
        >>> config["leaderboard_type"] == LeaderboardType.BIGCODE
        True

        >>> get_recommended_leaderboard_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty
    """
    if model_type is None:
        msg = "model_type cannot be None"
        raise ValueError(msg)

    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    # Base configuration
    base_config: dict[str, Any] = {
        "leaderboard_type": LeaderboardType.OPEN_LLM,
        "ranking_method": RankingMethod.SCORE,
        "metrics": ("mmlu", "hellaswag", "arc", "truthfulqa", "winogrande"),
        "higher_is_better": True,
        "precision": "float16",
        "num_few_shot": 5,
    }

    # Model-specific configurations
    model_configs: dict[str, dict[str, Any]] = {
        "llm": {
            "leaderboard_type": LeaderboardType.OPEN_LLM,
            "ranking_method": RankingMethod.WEIGHTED,
            "metrics": (
                "mmlu", "hellaswag", "arc", "truthfulqa", "winogrande", "gsm8k",
            ),
            "higher_is_better": True,
            "precision": "float16",
            "num_few_shot": 5,
        },
        "embedding": {
            "leaderboard_type": LeaderboardType.MTEB,
            "ranking_method": RankingMethod.SCORE,
            "metrics": (
                "classification",
                "clustering",
                "pair_classification",
                "reranking",
                "retrieval",
                "sts",
                "summarization",
            ),
            "higher_is_better": True,
            "precision": "float32",
            "num_few_shot": 0,
        },
        "code": {
            "leaderboard_type": LeaderboardType.BIGCODE,
            "ranking_method": RankingMethod.SCORE,
            "metrics": ("humaneval", "mbpp", "multipl_e"),
            "higher_is_better": True,
            "precision": "float16",
            "num_few_shot": 0,
        },
        "chat": {
            "leaderboard_type": LeaderboardType.CHATBOT_ARENA,
            "ranking_method": RankingMethod.ELO,
            "metrics": ("elo_rating", "win_rate", "votes"),
            "higher_is_better": True,
            "precision": "float16",
            "num_few_shot": 0,
        },
        "instruction": {
            "leaderboard_type": LeaderboardType.HELM,
            "ranking_method": RankingMethod.WEIGHTED,
            "metrics": ("accuracy", "robustness", "fairness", "efficiency"),
            "higher_is_better": True,
            "precision": "float16",
            "num_few_shot": 0,
        },
    }

    config = model_configs.get(model_type.lower(), base_config)

    # Apply task-specific adjustments
    if task_type:
        task_lower = task_type.lower()
        if task_lower == "zero_shot":
            config["num_few_shot"] = 0
        elif task_lower == "few_shot":
            config["num_few_shot"] = 5
        elif task_lower == "arena":
            config["leaderboard_type"] = LeaderboardType.CHATBOT_ARENA
            config["ranking_method"] = RankingMethod.ELO

    return config


def create_leaderboard_stats(
    leaderboard: Leaderboard,
) -> LeaderboardStats:
    """Create LeaderboardStats from a Leaderboard instance.

    Args:
        leaderboard: Leaderboard to compute stats for.

    Returns:
        LeaderboardStats instance with computed values.

    Raises:
        ValueError: If leaderboard is None.

    Examples:
        >>> config = LeaderboardConfig(name="test")
        >>> board = create_leaderboard(config)
        >>> entry = LeaderboardEntry("m1", 1, [ModelScore("a", 0.8)], "2024-01-01")
        >>> add_entry(board, entry)
        >>> stats = create_leaderboard_stats(board)
        >>> stats.total_models
        1
        >>> stats.avg_score
        0.8
        >>> stats.top_score
        0.8

        >>> create_leaderboard_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: leaderboard cannot be None
    """
    if leaderboard is None:
        msg = "leaderboard cannot be None"
        raise ValueError(msg)

    if not leaderboard.entries:
        return LeaderboardStats(
            total_models=0,
            avg_score=0.0,
            top_score=0.0,
            last_updated=datetime.now().isoformat(),
        )

    avg_scores = [compute_average_score(e) for e in leaderboard.entries]

    return LeaderboardStats(
        total_models=len(leaderboard.entries),
        avg_score=sum(avg_scores) / len(avg_scores),
        top_score=max(avg_scores),
        last_updated=datetime.now().isoformat(),
    )
