"""Tests for leaderboard functionality."""

from __future__ import annotations

import pytest

from hf_gtc.evaluation.leaderboards import (
    Leaderboard,
    LeaderboardCategory,
    LeaderboardConfig,
    LeaderboardEntry,
    ModelScore,
    SubmissionResult,
    SubmissionStatus,
    add_entry,
    compare_entries,
    compute_average_score,
    compute_leaderboard_stats,
    create_leaderboard,
    create_submission,
    filter_entries_by_size,
    find_entry_by_model,
    format_leaderboard,
    get_category,
    get_score_by_metric,
    get_top_entries,
    list_categories,
    list_submission_statuses,
    parse_submission_result,
    validate_category,
    validate_leaderboard_config,
    validate_submission_status,
)


class TestLeaderboardCategory:
    """Tests for LeaderboardCategory enum."""

    def test_llm_value(self) -> None:
        """Test LLM value."""
        assert LeaderboardCategory.LLM.value == "llm"

    def test_vision_value(self) -> None:
        """Test VISION value."""
        assert LeaderboardCategory.VISION.value == "vision"

    def test_speech_value(self) -> None:
        """Test SPEECH value."""
        assert LeaderboardCategory.SPEECH.value == "speech"

    def test_multimodal_value(self) -> None:
        """Test MULTIMODAL value."""
        assert LeaderboardCategory.MULTIMODAL.value == "multimodal"

    def test_custom_value(self) -> None:
        """Test CUSTOM value."""
        assert LeaderboardCategory.CUSTOM.value == "custom"


class TestSubmissionStatus:
    """Tests for SubmissionStatus enum."""

    def test_pending_value(self) -> None:
        """Test PENDING value."""
        assert SubmissionStatus.PENDING.value == "pending"

    def test_running_value(self) -> None:
        """Test RUNNING value."""
        assert SubmissionStatus.RUNNING.value == "running"

    def test_completed_value(self) -> None:
        """Test COMPLETED value."""
        assert SubmissionStatus.COMPLETED.value == "completed"

    def test_failed_value(self) -> None:
        """Test FAILED value."""
        assert SubmissionStatus.FAILED.value == "failed"

    def test_rejected_value(self) -> None:
        """Test REJECTED value."""
        assert SubmissionStatus.REJECTED.value == "rejected"


class TestLeaderboardConfig:
    """Tests for LeaderboardConfig dataclass."""

    def test_required_name(self) -> None:
        """Test that name is required."""
        config = LeaderboardConfig(name="test")
        assert config.name == "test"

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LeaderboardConfig(name="test")
        assert config.category == LeaderboardCategory.LLM
        assert config.url is None
        assert config.api_endpoint is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LeaderboardConfig(
            name="custom",
            category=LeaderboardCategory.VISION,
            url="https://example.com",
            api_endpoint="https://api.example.com",
        )
        assert config.category == LeaderboardCategory.VISION
        assert config.url == "https://example.com"

    def test_frozen(self) -> None:
        """Test that LeaderboardConfig is immutable."""
        config = LeaderboardConfig(name="test")
        with pytest.raises(AttributeError):
            config.name = "new"  # type: ignore[misc]


class TestModelScore:
    """Tests for ModelScore dataclass."""

    def test_creation(self) -> None:
        """Test creating ModelScore instance."""
        score = ModelScore("accuracy", 0.95)
        assert score.metric_name == "accuracy"
        assert score.score == 0.95
        assert score.is_higher_better is True

    def test_lower_is_better(self) -> None:
        """Test score where lower is better."""
        score = ModelScore("loss", 0.05, is_higher_better=False)
        assert score.is_higher_better is False

    def test_frozen(self) -> None:
        """Test that ModelScore is immutable."""
        score = ModelScore("test", 0.5)
        with pytest.raises(AttributeError):
            score.score = 0.9  # type: ignore[misc]


class TestLeaderboardEntry:
    """Tests for LeaderboardEntry dataclass."""

    def test_creation(self) -> None:
        """Test creating LeaderboardEntry instance."""
        entry = LeaderboardEntry(
            model_name="gpt-4",
            rank=1,
            scores=[ModelScore("accuracy", 0.95)],
            submission_date="2024-01-15",
        )
        assert entry.model_name == "gpt-4"
        assert entry.rank == 1
        assert len(entry.scores) == 1

    def test_with_model_size(self) -> None:
        """Test entry with model size."""
        entry = LeaderboardEntry(
            model_name="llama-7b",
            rank=5,
            scores=[],
            submission_date="2024-01-01",
            model_size=7000000000,
        )
        assert entry.model_size == 7000000000

    def test_frozen(self) -> None:
        """Test that LeaderboardEntry is immutable."""
        entry = LeaderboardEntry("test", 1, [], "2024-01-01")
        with pytest.raises(AttributeError):
            entry.rank = 2  # type: ignore[misc]


class TestSubmissionResult:
    """Tests for SubmissionResult dataclass."""

    def test_successful_result(self) -> None:
        """Test successful submission result."""
        result = SubmissionResult(
            submission_id="sub-001",
            status=SubmissionStatus.COMPLETED,
            scores=[ModelScore("acc", 0.92)],
            rank=5,
            error_message=None,
        )
        assert result.status == SubmissionStatus.COMPLETED
        assert result.rank == 5

    def test_failed_result(self) -> None:
        """Test failed submission result."""
        result = SubmissionResult(
            submission_id="sub-002",
            status=SubmissionStatus.FAILED,
            scores=None,
            rank=None,
            error_message="Evaluation failed",
        )
        assert result.status == SubmissionStatus.FAILED
        assert result.error_message == "Evaluation failed"


class TestLeaderboard:
    """Tests for Leaderboard dataclass."""

    def test_creation(self) -> None:
        """Test creating Leaderboard instance."""
        config = LeaderboardConfig(name="test")
        board = Leaderboard(config)
        assert board.config.name == "test"
        assert board.entries == []


class TestValidateLeaderboardConfig:
    """Tests for validate_leaderboard_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = LeaderboardConfig(name="test")
        validate_leaderboard_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_leaderboard_config(None)  # type: ignore[arg-type]

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        config = LeaderboardConfig(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_leaderboard_config(config)


class TestCreateLeaderboard:
    """Tests for create_leaderboard function."""

    def test_creates_leaderboard(self) -> None:
        """Test that function creates a leaderboard."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        assert isinstance(board, Leaderboard)
        assert board.config.name == "test"

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            create_leaderboard(None)  # type: ignore[arg-type]


class TestAddEntry:
    """Tests for add_entry function."""

    def test_adds_entry(self) -> None:
        """Test adding entry to leaderboard."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        entry = LeaderboardEntry("model", 1, [], "2024-01-01")
        add_entry(board, entry)
        assert len(board.entries) == 1

    def test_none_leaderboard_raises_error(self) -> None:
        """Test that None leaderboard raises ValueError."""
        entry = LeaderboardEntry("model", 1, [], "2024-01-01")
        with pytest.raises(ValueError, match="leaderboard cannot be None"):
            add_entry(None, entry)  # type: ignore[arg-type]

    def test_none_entry_raises_error(self) -> None:
        """Test that None entry raises ValueError."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        with pytest.raises(ValueError, match="entry cannot be None"):
            add_entry(board, None)  # type: ignore[arg-type]


class TestGetTopEntries:
    """Tests for get_top_entries function."""

    def test_gets_top_entries(self) -> None:
        """Test getting top entries."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        add_entry(board, LeaderboardEntry("m1", 3, [], "2024-01-01"))
        add_entry(board, LeaderboardEntry("m2", 1, [], "2024-01-01"))
        add_entry(board, LeaderboardEntry("m3", 2, [], "2024-01-01"))

        top = get_top_entries(board, 2)
        assert len(top) == 2
        assert top[0].model_name == "m2"
        assert top[1].model_name == "m3"

    def test_empty_leaderboard(self) -> None:
        """Test with empty leaderboard."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        top = get_top_entries(board, 5)
        assert top == []

    def test_none_leaderboard_raises_error(self) -> None:
        """Test that None leaderboard raises ValueError."""
        with pytest.raises(ValueError, match="leaderboard cannot be None"):
            get_top_entries(None, 5)  # type: ignore[arg-type]

    def test_non_positive_n_raises_error(self) -> None:
        """Test that non-positive n raises ValueError."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        with pytest.raises(ValueError, match="n must be positive"):
            get_top_entries(board, 0)


class TestFindEntryByModel:
    """Tests for find_entry_by_model function."""

    def test_finds_entry(self) -> None:
        """Test finding existing entry."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        entry = LeaderboardEntry("my-model", 1, [], "2024-01-01")
        add_entry(board, entry)

        found = find_entry_by_model(board, "my-model")
        assert found is not None
        assert found.rank == 1

    def test_not_found(self) -> None:
        """Test when entry not found."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        found = find_entry_by_model(board, "unknown")
        assert found is None

    def test_none_leaderboard_raises_error(self) -> None:
        """Test that None leaderboard raises ValueError."""
        with pytest.raises(ValueError, match="leaderboard cannot be None"):
            find_entry_by_model(None, "test")  # type: ignore[arg-type]

    def test_empty_model_name_raises_error(self) -> None:
        """Test that empty model_name raises ValueError."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            find_entry_by_model(board, "")


class TestComputeAverageScore:
    """Tests for compute_average_score function."""

    def test_computes_average(self) -> None:
        """Test computing average score."""
        scores = [ModelScore("a", 0.8), ModelScore("b", 0.9)]
        entry = LeaderboardEntry("model", 1, scores, "2024-01-01")
        avg = compute_average_score(entry)
        assert avg == pytest.approx(0.85)

    def test_empty_scores(self) -> None:
        """Test with no scores."""
        entry = LeaderboardEntry("model", 1, [], "2024-01-01")
        avg = compute_average_score(entry)
        assert avg == 0.0

    def test_single_score(self) -> None:
        """Test with single score."""
        entry = LeaderboardEntry("model", 1, [ModelScore("a", 0.95)], "2024-01-01")
        avg = compute_average_score(entry)
        assert avg == 0.95

    def test_none_entry_raises_error(self) -> None:
        """Test that None entry raises ValueError."""
        with pytest.raises(ValueError, match="entry cannot be None"):
            compute_average_score(None)  # type: ignore[arg-type]


class TestGetScoreByMetric:
    """Tests for get_score_by_metric function."""

    def test_finds_score(self) -> None:
        """Test finding existing score."""
        scores = [ModelScore("accuracy", 0.95), ModelScore("f1", 0.92)]
        entry = LeaderboardEntry("model", 1, scores, "2024-01-01")
        score = get_score_by_metric(entry, "accuracy")
        assert score == 0.95

    def test_not_found(self) -> None:
        """Test when metric not found."""
        entry = LeaderboardEntry("model", 1, [], "2024-01-01")
        score = get_score_by_metric(entry, "unknown")
        assert score is None

    def test_none_entry_raises_error(self) -> None:
        """Test that None entry raises ValueError."""
        with pytest.raises(ValueError, match="entry cannot be None"):
            get_score_by_metric(None, "test")  # type: ignore[arg-type]

    def test_empty_metric_name_raises_error(self) -> None:
        """Test that empty metric_name raises ValueError."""
        entry = LeaderboardEntry("model", 1, [], "2024-01-01")
        with pytest.raises(ValueError, match="metric_name cannot be empty"):
            get_score_by_metric(entry, "")


class TestCreateSubmission:
    """Tests for create_submission function."""

    def test_creates_submission(self) -> None:
        """Test creating submission payload."""
        payload = create_submission("my-model", "org/my-model")
        assert payload["model_name"] == "my-model"
        assert payload["model_path"] == "org/my-model"
        assert "submission_date" in payload

    def test_with_size(self) -> None:
        """Test with model size."""
        payload = create_submission("model", "path", model_size=7000000000)
        assert payload["model_size"] == 7000000000

    def test_with_metadata(self) -> None:
        """Test with metadata."""
        payload = create_submission("model", "path", metadata={"version": "1.0"})
        assert payload["metadata"]["version"] == "1.0"

    def test_empty_model_name_raises_error(self) -> None:
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_submission("", "path")

    def test_empty_model_path_raises_error(self) -> None:
        """Test that empty model_path raises ValueError."""
        with pytest.raises(ValueError, match="model_path cannot be empty"):
            create_submission("model", "")


class TestParseSubmissionResult:
    """Tests for parse_submission_result function."""

    def test_parses_completed(self) -> None:
        """Test parsing completed submission."""
        response = {
            "submission_id": "sub-001",
            "status": "completed",
            "scores": [{"metric_name": "acc", "score": 0.9}],
            "rank": 5,
        }
        result = parse_submission_result(response)
        assert result.status == SubmissionStatus.COMPLETED
        assert result.rank == 5
        assert len(result.scores or []) == 1

    def test_parses_failed(self) -> None:
        """Test parsing failed submission."""
        response = {
            "submission_id": "sub-002",
            "status": "failed",
            "error_message": "Error",
        }
        result = parse_submission_result(response)
        assert result.status == SubmissionStatus.FAILED
        assert result.error_message == "Error"

    def test_none_response_raises_error(self) -> None:
        """Test that None response raises ValueError."""
        with pytest.raises(ValueError, match="response cannot be None"):
            parse_submission_result(None)  # type: ignore[arg-type]

    def test_missing_id_raises_error(self) -> None:
        """Test that missing submission_id raises ValueError."""
        with pytest.raises(ValueError, match="missing required field"):
            parse_submission_result({"status": "completed"})

    def test_missing_status_raises_error(self) -> None:
        """Test that missing status raises ValueError."""
        with pytest.raises(ValueError, match="missing required field"):
            parse_submission_result({"submission_id": "001"})


class TestCompareEntries:
    """Tests for compare_entries function."""

    def test_compare_entries(self) -> None:
        """Test comparing two entries."""
        e1 = LeaderboardEntry("m1", 1, [ModelScore("acc", 0.9)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 2, [ModelScore("acc", 0.85)], "2024-01-02")
        cmp = compare_entries(e1, e2)
        assert cmp["rank_difference"] == -1
        assert cmp["better_ranked"] == "m1"
        assert cmp["higher_score"] == "m1"

    def test_none_entry1_raises_error(self) -> None:
        """Test that None entry1 raises ValueError."""
        e2 = LeaderboardEntry("m2", 2, [], "2024-01-01")
        with pytest.raises(ValueError, match="entry1 cannot be None"):
            compare_entries(None, e2)  # type: ignore[arg-type]

    def test_none_entry2_raises_error(self) -> None:
        """Test that None entry2 raises ValueError."""
        e1 = LeaderboardEntry("m1", 1, [], "2024-01-01")
        with pytest.raises(ValueError, match="entry2 cannot be None"):
            compare_entries(e1, None)  # type: ignore[arg-type]


class TestFilterEntriesBySize:
    """Tests for filter_entries_by_size function."""

    def test_filter_by_max_size(self) -> None:
        """Test filtering by max size."""
        e1 = LeaderboardEntry("m1", 1, [], "2024-01-01", model_size=7000000000)
        e2 = LeaderboardEntry("m2", 2, [], "2024-01-01", model_size=13000000000)
        filtered = filter_entries_by_size([e1, e2], max_size=10000000000)
        assert len(filtered) == 1
        assert filtered[0].model_name == "m1"

    def test_filter_by_min_size(self) -> None:
        """Test filtering by min size."""
        e1 = LeaderboardEntry("m1", 1, [], "2024-01-01", model_size=7000000000)
        e2 = LeaderboardEntry("m2", 2, [], "2024-01-01", model_size=13000000000)
        filtered = filter_entries_by_size([e1, e2], min_size=10000000000)
        assert len(filtered) == 1
        assert filtered[0].model_name == "m2"

    def test_skips_entries_without_size(self) -> None:
        """Test that entries without size are skipped."""
        e1 = LeaderboardEntry("m1", 1, [], "2024-01-01")
        filtered = filter_entries_by_size([e1], max_size=10000000000)
        assert len(filtered) == 0

    def test_none_entries_raises_error(self) -> None:
        """Test that None entries raises ValueError."""
        with pytest.raises(ValueError, match="entries cannot be None"):
            filter_entries_by_size(None)  # type: ignore[arg-type]


class TestListCategories:
    """Tests for list_categories function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        categories = list_categories()
        assert isinstance(categories, list)

    def test_contains_expected_categories(self) -> None:
        """Test that list contains expected categories."""
        categories = list_categories()
        assert "llm" in categories
        assert "vision" in categories

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        categories = list_categories()
        assert categories == sorted(categories)


class TestValidateCategory:
    """Tests for validate_category function."""

    def test_valid_llm(self) -> None:
        """Test validation of llm category."""
        assert validate_category("llm") is True

    def test_valid_vision(self) -> None:
        """Test validation of vision category."""
        assert validate_category("vision") is True

    def test_invalid_category(self) -> None:
        """Test validation of invalid category."""
        assert validate_category("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_category("") is False


class TestGetCategory:
    """Tests for get_category function."""

    def test_get_llm(self) -> None:
        """Test getting LLM category."""
        result = get_category("llm")
        assert result == LeaderboardCategory.LLM

    def test_get_vision(self) -> None:
        """Test getting VISION category."""
        result = get_category("vision")
        assert result == LeaderboardCategory.VISION

    def test_invalid_category_raises_error(self) -> None:
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="invalid category"):
            get_category("invalid")


class TestListSubmissionStatuses:
    """Tests for list_submission_statuses function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        statuses = list_submission_statuses()
        assert isinstance(statuses, list)

    def test_contains_expected_statuses(self) -> None:
        """Test that list contains expected statuses."""
        statuses = list_submission_statuses()
        assert "completed" in statuses
        assert "pending" in statuses

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        statuses = list_submission_statuses()
        assert statuses == sorted(statuses)


class TestValidateSubmissionStatus:
    """Tests for validate_submission_status function."""

    def test_valid_completed(self) -> None:
        """Test validation of completed status."""
        assert validate_submission_status("completed") is True

    def test_valid_pending(self) -> None:
        """Test validation of pending status."""
        assert validate_submission_status("pending") is True

    def test_invalid_status(self) -> None:
        """Test validation of invalid status."""
        assert validate_submission_status("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_submission_status("") is False


class TestFormatLeaderboard:
    """Tests for format_leaderboard function."""

    def test_format_empty(self) -> None:
        """Test formatting empty leaderboard."""
        config = LeaderboardConfig(name="Test Board")
        board = create_leaderboard(config)
        formatted = format_leaderboard(board)
        assert "Test Board" in formatted
        assert "No entries" in formatted

    def test_format_with_entries(self) -> None:
        """Test formatting leaderboard with entries."""
        config = LeaderboardConfig(name="Test")
        board = create_leaderboard(config)
        entry = LeaderboardEntry("m1", 1, [ModelScore("a", 0.9)], "2024-01-01")
        add_entry(board, entry)
        formatted = format_leaderboard(board)
        assert "m1" in formatted

    def test_none_leaderboard_raises_error(self) -> None:
        """Test that None leaderboard raises ValueError."""
        with pytest.raises(ValueError, match="leaderboard cannot be None"):
            format_leaderboard(None)  # type: ignore[arg-type]


class TestComputeLeaderboardStats:
    """Tests for compute_leaderboard_stats function."""

    def test_empty_leaderboard(self) -> None:
        """Test stats for empty leaderboard."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        stats = compute_leaderboard_stats(board)
        assert stats["total_entries"] == 0
        assert stats["avg_score"] == 0.0

    def test_with_entries(self) -> None:
        """Test stats with entries."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        e1 = LeaderboardEntry("m1", 1, [ModelScore("a", 0.8)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 2, [ModelScore("a", 0.9)], "2024-01-01")
        add_entry(board, e1)
        add_entry(board, e2)
        stats = compute_leaderboard_stats(board)
        assert stats["total_entries"] == 2
        assert stats["avg_score"] == pytest.approx(0.85)

    def test_none_leaderboard_raises_error(self) -> None:
        """Test that None leaderboard raises ValueError."""
        with pytest.raises(ValueError, match="leaderboard cannot be None"):
            compute_leaderboard_stats(None)  # type: ignore[arg-type]
