"""Tests for leaderboard functionality."""

from __future__ import annotations

import pytest

from hf_gtc.evaluation.leaderboards import (
    VALID_LEADERBOARD_TYPES,
    VALID_RANKING_METHODS,
    Leaderboard,
    LeaderboardCategory,
    LeaderboardConfig,
    LeaderboardEntry,
    LeaderboardStats,
    LeaderboardType,
    ModelScore,
    RankingMethod,
    SubmissionConfig,
    SubmissionResult,
    SubmissionStatus,
    add_entry,
    calculate_ranking,
    compare_entries,
    compare_models,
    compute_average_score,
    compute_elo_rating,
    compute_leaderboard_stats,
    create_leaderboard,
    create_leaderboard_config,
    create_leaderboard_entry,
    create_leaderboard_stats,
    create_submission,
    create_submission_config,
    filter_entries_by_size,
    find_entry_by_model,
    format_leaderboard,
    format_leaderboard_stats,
    format_submission,
    get_category,
    get_leaderboard_type,
    get_ranking_method,
    get_recommended_leaderboard_config,
    get_score_by_metric,
    get_top_entries,
    list_categories,
    list_leaderboard_types,
    list_ranking_methods,
    list_submission_statuses,
    parse_submission_result,
    validate_category,
    validate_leaderboard_config,
    validate_leaderboard_stats,
    validate_leaderboard_type,
    validate_ranking_method,
    validate_submission,
    validate_submission_config,
    validate_submission_status,
)


class TestLeaderboardType:
    """Tests for LeaderboardType enum."""

    def test_open_llm_value(self) -> None:
        """Test OPEN_LLM value."""
        assert LeaderboardType.OPEN_LLM.value == "open_llm"

    def test_mteb_value(self) -> None:
        """Test MTEB value."""
        assert LeaderboardType.MTEB.value == "mteb"

    def test_helm_value(self) -> None:
        """Test HELM value."""
        assert LeaderboardType.HELM.value == "helm"

    def test_bigcode_value(self) -> None:
        """Test BIGCODE value."""
        assert LeaderboardType.BIGCODE.value == "bigcode"

    def test_chatbot_arena_value(self) -> None:
        """Test CHATBOT_ARENA value."""
        assert LeaderboardType.CHATBOT_ARENA.value == "chatbot_arena"


class TestRankingMethod:
    """Tests for RankingMethod enum."""

    def test_score_value(self) -> None:
        """Test SCORE value."""
        assert RankingMethod.SCORE.value == "score"

    def test_elo_value(self) -> None:
        """Test ELO value."""
        assert RankingMethod.ELO.value == "elo"

    def test_win_rate_value(self) -> None:
        """Test WIN_RATE value."""
        assert RankingMethod.WIN_RATE.value == "win_rate"

    def test_weighted_value(self) -> None:
        """Test WEIGHTED value."""
        assert RankingMethod.WEIGHTED.value == "weighted"


class TestValidFrozensets:
    """Tests for VALID_* frozensets."""

    def test_valid_leaderboard_types(self) -> None:
        """Test VALID_LEADERBOARD_TYPES contains all enum values."""
        assert "open_llm" in VALID_LEADERBOARD_TYPES
        assert "mteb" in VALID_LEADERBOARD_TYPES
        assert "helm" in VALID_LEADERBOARD_TYPES
        assert "bigcode" in VALID_LEADERBOARD_TYPES
        assert "chatbot_arena" in VALID_LEADERBOARD_TYPES

    def test_valid_ranking_methods(self) -> None:
        """Test VALID_RANKING_METHODS contains all enum values."""
        assert "score" in VALID_RANKING_METHODS
        assert "elo" in VALID_RANKING_METHODS
        assert "win_rate" in VALID_RANKING_METHODS
        assert "weighted" in VALID_RANKING_METHODS


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
        assert score.score == pytest.approx(0.95)
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
        assert avg == pytest.approx(0.0)

    def test_single_score(self) -> None:
        """Test with single score."""
        entry = LeaderboardEntry("model", 1, [ModelScore("a", 0.95)], "2024-01-01")
        avg = compute_average_score(entry)
        assert avg == pytest.approx(0.95)

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
        assert score == pytest.approx(0.95)

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
        assert stats["avg_score"] == pytest.approx(0.0)

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


class TestLeaderboardConfigExtended:
    """Tests for extended LeaderboardConfig functionality."""

    def test_new_default_values(self) -> None:
        """Test new default configuration values."""
        config = LeaderboardConfig(name="test")
        assert config.leaderboard_type == LeaderboardType.OPEN_LLM
        assert config.metrics == ()
        assert config.ranking_method == RankingMethod.SCORE
        assert config.higher_is_better is True

    def test_custom_new_values(self) -> None:
        """Test custom new configuration values."""
        config = LeaderboardConfig(
            name="custom",
            leaderboard_type=LeaderboardType.CHATBOT_ARENA,
            metrics=("elo", "win_rate"),
            ranking_method=RankingMethod.ELO,
            higher_is_better=True,
        )
        assert config.leaderboard_type == LeaderboardType.CHATBOT_ARENA
        assert config.metrics == ("elo", "win_rate")
        assert config.ranking_method == RankingMethod.ELO


class TestSubmissionConfig:
    """Tests for SubmissionConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating SubmissionConfig instance."""
        config = SubmissionConfig(model_name="meta-llama/Llama-2-7b")
        assert config.model_name == "meta-llama/Llama-2-7b"

    def test_default_values(self) -> None:
        """Test default values."""
        config = SubmissionConfig(model_name="test")
        assert config.revision == "main"
        assert config.precision == "float16"
        assert config.num_few_shot == 0

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = SubmissionConfig(
            model_name="test",
            revision="v1.0",
            precision="bfloat16",
            num_few_shot=5,
        )
        assert config.revision == "v1.0"
        assert config.precision == "bfloat16"
        assert config.num_few_shot == 5

    def test_frozen(self) -> None:
        """Test that SubmissionConfig is immutable."""
        config = SubmissionConfig(model_name="test")
        with pytest.raises(AttributeError):
            config.model_name = "new"  # type: ignore[misc]


class TestLeaderboardStats:
    """Tests for LeaderboardStats dataclass."""

    def test_creation(self) -> None:
        """Test creating LeaderboardStats instance."""
        stats = LeaderboardStats(
            total_models=100,
            avg_score=0.65,
            top_score=0.89,
            last_updated="2024-01-15T12:00:00",
        )
        assert stats.total_models == 100
        assert stats.avg_score == pytest.approx(0.65)
        assert stats.top_score == pytest.approx(0.89)

    def test_frozen(self) -> None:
        """Test that LeaderboardStats is immutable."""
        stats = LeaderboardStats(100, 0.65, 0.89, "2024-01-15")
        with pytest.raises(AttributeError):
            stats.total_models = 200  # type: ignore[misc]


class TestListLeaderboardTypes:
    """Tests for list_leaderboard_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_leaderboard_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_leaderboard_types()
        assert "open_llm" in types
        assert "mteb" in types
        assert "chatbot_arena" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_leaderboard_types()
        assert types == sorted(types)


class TestGetLeaderboardType:
    """Tests for get_leaderboard_type function."""

    def test_get_open_llm(self) -> None:
        """Test getting OPEN_LLM type."""
        result = get_leaderboard_type("open_llm")
        assert result == LeaderboardType.OPEN_LLM

    def test_get_chatbot_arena(self) -> None:
        """Test getting CHATBOT_ARENA type."""
        result = get_leaderboard_type("chatbot_arena")
        assert result == LeaderboardType.CHATBOT_ARENA

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid leaderboard type"):
            get_leaderboard_type("invalid")


class TestValidateLeaderboardType:
    """Tests for validate_leaderboard_type function."""

    def test_valid_open_llm(self) -> None:
        """Test validation of open_llm type."""
        assert validate_leaderboard_type("open_llm") is True

    def test_valid_mteb(self) -> None:
        """Test validation of mteb type."""
        assert validate_leaderboard_type("mteb") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_leaderboard_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_leaderboard_type("") is False


class TestListRankingMethods:
    """Tests for list_ranking_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_ranking_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_ranking_methods()
        assert "score" in methods
        assert "elo" in methods
        assert "win_rate" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_ranking_methods()
        assert methods == sorted(methods)


class TestGetRankingMethod:
    """Tests for get_ranking_method function."""

    def test_get_score(self) -> None:
        """Test getting SCORE method."""
        result = get_ranking_method("score")
        assert result == RankingMethod.SCORE

    def test_get_elo(self) -> None:
        """Test getting ELO method."""
        result = get_ranking_method("elo")
        assert result == RankingMethod.ELO

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid ranking method"):
            get_ranking_method("invalid")


class TestValidateRankingMethod:
    """Tests for validate_ranking_method function."""

    def test_valid_score(self) -> None:
        """Test validation of score method."""
        assert validate_ranking_method("score") is True

    def test_valid_elo(self) -> None:
        """Test validation of elo method."""
        assert validate_ranking_method("elo") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_ranking_method("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_ranking_method("") is False


class TestCreateLeaderboardConfig:
    """Tests for create_leaderboard_config function."""

    def test_creates_config(self) -> None:
        """Test creating a leaderboard config."""
        config = create_leaderboard_config("my-leaderboard")
        assert config.name == "my-leaderboard"

    def test_with_options(self) -> None:
        """Test creating config with options."""
        config = create_leaderboard_config(
            "Open LLM",
            leaderboard_type=LeaderboardType.OPEN_LLM,
            metrics=("mmlu", "hellaswag"),
            ranking_method=RankingMethod.WEIGHTED,
        )
        assert config.leaderboard_type == LeaderboardType.OPEN_LLM
        assert config.metrics == ("mmlu", "hellaswag")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_leaderboard_config("")


class TestCreateSubmissionConfig:
    """Tests for create_submission_config function."""

    def test_creates_config(self) -> None:
        """Test creating a submission config."""
        config = create_submission_config("meta-llama/Llama-2-7b")
        assert config.model_name == "meta-llama/Llama-2-7b"

    def test_with_options(self) -> None:
        """Test creating config with options."""
        config = create_submission_config(
            "mistralai/Mistral-7B",
            revision="v0.1",
            precision="bfloat16",
            num_few_shot=5,
        )
        assert config.revision == "v0.1"
        assert config.num_few_shot == 5

    def test_empty_model_name_raises_error(self) -> None:
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_submission_config("")

    def test_negative_few_shot_raises_error(self) -> None:
        """Test that negative num_few_shot raises ValueError."""
        with pytest.raises(ValueError, match="num_few_shot cannot be negative"):
            create_submission_config("test", num_few_shot=-1)


class TestCreateLeaderboardEntry:
    """Tests for create_leaderboard_entry function."""

    def test_creates_entry(self) -> None:
        """Test creating a leaderboard entry."""
        entry = create_leaderboard_entry(
            "gpt-4",
            {"accuracy": 0.95, "f1": 0.92},
            1,
            "2024-01-15",
        )
        assert entry.model_name == "gpt-4"
        assert entry.rank == 1
        assert len(entry.scores) == 2

    def test_with_model_size(self) -> None:
        """Test creating entry with model size."""
        entry = create_leaderboard_entry(
            "llama-7b",
            {"mmlu": 0.65},
            5,
            "2024-01-10",
            model_size=7000000000,
        )
        assert entry.model_size == 7000000000

    def test_empty_model_name_raises_error(self) -> None:
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_leaderboard_entry("", {}, 1, "2024-01-01")

    def test_non_positive_rank_raises_error(self) -> None:
        """Test that non-positive rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            create_leaderboard_entry("test", {}, 0, "2024-01-01")


class TestValidateSubmissionConfig:
    """Tests for validate_submission_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = SubmissionConfig(model_name="test-model")
        validate_submission_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_submission_config(None)  # type: ignore[arg-type]

    def test_empty_model_name_raises_error(self) -> None:
        """Test that empty model_name raises ValueError."""
        config = SubmissionConfig(model_name="")
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            validate_submission_config(config)

    def test_negative_few_shot_raises_error(self) -> None:
        """Test that negative num_few_shot raises ValueError."""
        config = SubmissionConfig(model_name="test", num_few_shot=-1)
        with pytest.raises(ValueError, match="num_few_shot cannot be negative"):
            validate_submission_config(config)


class TestValidateLeaderboardStats:
    """Tests for validate_leaderboard_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = LeaderboardStats(100, 0.65, 0.89, "2024-01-15")
        validate_leaderboard_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_leaderboard_stats(None)  # type: ignore[arg-type]

    def test_negative_total_models_raises_error(self) -> None:
        """Test that negative total_models raises ValueError."""
        stats = LeaderboardStats(-1, 0.65, 0.89, "2024-01-15")
        with pytest.raises(ValueError, match="total_models cannot be negative"):
            validate_leaderboard_stats(stats)

    def test_invalid_avg_score_raises_error(self) -> None:
        """Test that invalid avg_score raises ValueError."""
        stats = LeaderboardStats(100, 1.5, 0.89, "2024-01-15")
        with pytest.raises(ValueError, match="avg_score must be between 0 and 1"):
            validate_leaderboard_stats(stats)

    def test_invalid_top_score_raises_error(self) -> None:
        """Test that invalid top_score raises ValueError."""
        stats = LeaderboardStats(100, 0.65, -0.1, "2024-01-15")
        with pytest.raises(ValueError, match="top_score must be between 0 and 1"):
            validate_leaderboard_stats(stats)


class TestValidateSubmission:
    """Tests for validate_submission function."""

    def test_valid_submission(self) -> None:
        """Test validation of valid submission."""
        submission = {"model_name": "test", "model_path": "org/test"}
        assert validate_submission(submission) is True

    def test_missing_model_name(self) -> None:
        """Test submission with missing model_name."""
        submission = {"model_path": "org/test"}
        assert validate_submission(submission) is False

    def test_empty_model_name(self) -> None:
        """Test submission with empty model_name."""
        submission = {"model_name": "", "model_path": "org/test"}
        assert validate_submission(submission) is False

    def test_missing_model_path(self) -> None:
        """Test submission with missing model_path."""
        submission = {"model_name": "test"}
        assert validate_submission(submission) is False

    def test_none_submission_raises_error(self) -> None:
        """Test that None submission raises ValueError."""
        with pytest.raises(ValueError, match="submission cannot be None"):
            validate_submission(None)  # type: ignore[arg-type]


class TestCalculateRanking:
    """Tests for calculate_ranking function."""

    def test_ranks_by_score(self) -> None:
        """Test ranking entries by score."""
        e1 = LeaderboardEntry("m1", 0, [ModelScore("a", 0.8)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 0, [ModelScore("a", 0.9)], "2024-01-01")
        ranked = calculate_ranking([e1, e2])
        assert ranked[0].model_name == "m2"
        assert ranked[0].rank == 1
        assert ranked[1].model_name == "m1"
        assert ranked[1].rank == 2

    def test_lower_is_better(self) -> None:
        """Test ranking when lower is better."""
        e1 = LeaderboardEntry("m1", 0, [ModelScore("loss", 0.1)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 0, [ModelScore("loss", 0.2)], "2024-01-01")
        ranked = calculate_ranking([e1, e2], higher_is_better=False)
        assert ranked[0].model_name == "m1"

    def test_empty_entries(self) -> None:
        """Test with empty entries list."""
        ranked = calculate_ranking([])
        assert ranked == []

    def test_none_entries_raises_error(self) -> None:
        """Test that None entries raises ValueError."""
        with pytest.raises(ValueError, match="entries cannot be None"):
            calculate_ranking(None)  # type: ignore[arg-type]


class TestComputeEloRating:
    """Tests for compute_elo_rating function."""

    def test_win_against_equal(self) -> None:
        """Test winning against equal opponent."""
        new_rating = compute_elo_rating(1500.0, 1500.0, 1.0)
        assert new_rating > 1500.0

    def test_loss_against_equal(self) -> None:
        """Test losing against equal opponent."""
        new_rating = compute_elo_rating(1500.0, 1500.0, 0.0)
        assert new_rating < 1500.0

    def test_draw_against_equal(self) -> None:
        """Test drawing against equal opponent."""
        new_rating = compute_elo_rating(1500.0, 1500.0, 0.5)
        assert new_rating == pytest.approx(1500.0)

    def test_win_against_stronger(self) -> None:
        """Test winning against stronger opponent."""
        new_rating = compute_elo_rating(1400.0, 1600.0, 1.0)
        gain = new_rating - 1400.0
        assert gain > 16.0  # More than half of k_factor

    def test_custom_k_factor(self) -> None:
        """Test with custom k_factor."""
        new_rating = compute_elo_rating(1500.0, 1500.0, 1.0, k_factor=16.0)
        assert new_rating == pytest.approx(1508.0)

    def test_invalid_outcome_raises_error(self) -> None:
        """Test that invalid outcome raises ValueError."""
        with pytest.raises(ValueError, match="outcome must be between 0 and 1"):
            compute_elo_rating(1500.0, 1500.0, 1.5)

    def test_invalid_k_factor_raises_error(self) -> None:
        """Test that invalid k_factor raises ValueError."""
        with pytest.raises(ValueError, match="k_factor must be positive"):
            compute_elo_rating(1500.0, 1500.0, 1.0, k_factor=-1.0)


class TestFormatSubmission:
    """Tests for format_submission function."""

    def test_formats_submission(self) -> None:
        """Test formatting submission config."""
        config = SubmissionConfig(
            model_name="meta-llama/Llama-2-7b",
            revision="main",
            precision="float16",
            num_few_shot=5,
        )
        formatted = format_submission(config)
        assert "meta-llama/Llama-2-7b" in formatted
        assert "float16" in formatted
        assert "5" in formatted

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="submission_config cannot be None"):
            format_submission(None)  # type: ignore[arg-type]


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compares_models(self) -> None:
        """Test comparing two models."""
        e1 = LeaderboardEntry("m1", 1, [ModelScore("acc", 0.9)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 2, [ModelScore("acc", 0.85)], "2024-01-02")
        cmp = compare_models(e1, e2)
        assert cmp["winner"] == "m1"
        assert cmp["rank_difference"] == -1
        assert cmp["score_difference"] > 0

    def test_with_specific_metrics(self) -> None:
        """Test comparing with specific metrics."""
        e1 = LeaderboardEntry(
            "m1", 1, [ModelScore("acc", 0.9), ModelScore("f1", 0.85)], "2024-01-01"
        )
        e2 = LeaderboardEntry(
            "m2", 2, [ModelScore("acc", 0.85), ModelScore("f1", 0.9)], "2024-01-02"
        )
        cmp = compare_models(e1, e2, metrics=["acc", "f1"])
        assert "acc" in cmp["per_metric"]
        assert "f1" in cmp["per_metric"]
        assert cmp["per_metric"]["acc"]["winner"] == "m1"
        assert cmp["per_metric"]["f1"]["winner"] == "m2"

    def test_tie(self) -> None:
        """Test when models have equal scores."""
        e1 = LeaderboardEntry("m1", 1, [ModelScore("acc", 0.9)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 2, [ModelScore("acc", 0.9)], "2024-01-02")
        cmp = compare_models(e1, e2)
        assert cmp["winner"] == "tie"

    def test_second_model_wins(self) -> None:
        """Test when second model has higher score."""
        e1 = LeaderboardEntry("m1", 1, [ModelScore("acc", 0.8)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 2, [ModelScore("acc", 0.9)], "2024-01-02")
        cmp = compare_models(e1, e2)
        assert cmp["winner"] == "m2"

    def test_none_model_a_raises_error(self) -> None:
        """Test that None model_a raises ValueError."""
        e2 = LeaderboardEntry("m2", 2, [], "2024-01-01")
        with pytest.raises(ValueError, match="model_a cannot be None"):
            compare_models(None, e2)  # type: ignore[arg-type]

    def test_none_model_b_raises_error(self) -> None:
        """Test that None model_b raises ValueError."""
        e1 = LeaderboardEntry("m1", 1, [], "2024-01-01")
        with pytest.raises(ValueError, match="model_b cannot be None"):
            compare_models(e1, None)  # type: ignore[arg-type]


class TestFormatLeaderboardStats:
    """Tests for format_leaderboard_stats function."""

    def test_formats_stats(self) -> None:
        """Test formatting leaderboard stats."""
        stats = LeaderboardStats(
            total_models=100,
            avg_score=0.65,
            top_score=0.89,
            last_updated="2024-01-15T12:00:00",
        )
        formatted = format_leaderboard_stats(stats)
        assert "100" in formatted
        assert "0.65" in formatted
        assert "0.89" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_leaderboard_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedLeaderboardConfig:
    """Tests for get_recommended_leaderboard_config function."""

    def test_llm_config(self) -> None:
        """Test config for LLM models."""
        config = get_recommended_leaderboard_config("llm")
        assert config["leaderboard_type"] == LeaderboardType.OPEN_LLM

    def test_embedding_config(self) -> None:
        """Test config for embedding models."""
        config = get_recommended_leaderboard_config("embedding")
        assert config["leaderboard_type"] == LeaderboardType.MTEB

    def test_code_config(self) -> None:
        """Test config for code models."""
        config = get_recommended_leaderboard_config("code")
        assert config["leaderboard_type"] == LeaderboardType.BIGCODE

    def test_chat_config(self) -> None:
        """Test config for chat models."""
        config = get_recommended_leaderboard_config("chat")
        assert config["leaderboard_type"] == LeaderboardType.CHATBOT_ARENA
        assert config["ranking_method"] == RankingMethod.ELO

    def test_with_task_type_zero_shot(self) -> None:
        """Test config with zero_shot task type."""
        config = get_recommended_leaderboard_config("llm", task_type="zero_shot")
        assert config["num_few_shot"] == 0

    def test_with_task_type_few_shot(self) -> None:
        """Test config with few_shot task type."""
        config = get_recommended_leaderboard_config("llm", task_type="few_shot")
        assert config["num_few_shot"] == 5

    def test_with_task_type_arena(self) -> None:
        """Test config with arena task type."""
        config = get_recommended_leaderboard_config("llm", task_type="arena")
        assert config["leaderboard_type"] == LeaderboardType.CHATBOT_ARENA
        assert config["ranking_method"] == RankingMethod.ELO

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_leaderboard_config("")

    def test_none_model_type_raises_error(self) -> None:
        """Test that None model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be None"):
            get_recommended_leaderboard_config(None)  # type: ignore[arg-type]


class TestCreateLeaderboardStats:
    """Tests for create_leaderboard_stats function."""

    def test_creates_stats(self) -> None:
        """Test creating leaderboard stats."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        entry = LeaderboardEntry("m1", 1, [ModelScore("a", 0.8)], "2024-01-01")
        add_entry(board, entry)
        stats = create_leaderboard_stats(board)
        assert stats.total_models == 1
        assert stats.avg_score == pytest.approx(0.8)
        assert stats.top_score == pytest.approx(0.8)

    def test_empty_leaderboard(self) -> None:
        """Test with empty leaderboard."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        stats = create_leaderboard_stats(board)
        assert stats.total_models == 0
        assert stats.avg_score == pytest.approx(0.0)
        assert stats.top_score == pytest.approx(0.0)

    def test_multiple_entries(self) -> None:
        """Test with multiple entries."""
        config = LeaderboardConfig(name="test")
        board = create_leaderboard(config)
        e1 = LeaderboardEntry("m1", 1, [ModelScore("a", 0.8)], "2024-01-01")
        e2 = LeaderboardEntry("m2", 2, [ModelScore("a", 0.9)], "2024-01-02")
        add_entry(board, e1)
        add_entry(board, e2)
        stats = create_leaderboard_stats(board)
        assert stats.total_models == 2
        assert stats.avg_score == pytest.approx(0.85)
        assert stats.top_score == pytest.approx(0.9)

    def test_none_leaderboard_raises_error(self) -> None:
        """Test that None leaderboard raises ValueError."""
        with pytest.raises(ValueError, match="leaderboard cannot be None"):
            create_leaderboard_stats(None)  # type: ignore[arg-type]
