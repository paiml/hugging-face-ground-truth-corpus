"""Tests for training.pruning module."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hf_gtc.training.pruning import (
    VALID_PRUNING_METHODS,
    VALID_PRUNING_SCHEDULES,
    VALID_PRUNING_SCOPES,
    IterativePruningConfig,
    LotteryTicketConfig,
    PruningConfig,
    PruningMethod,
    PruningSchedule,
    PruningScope,
    PruningStats,
    calculate_pruning_mask,
    calculate_sparsity,
    create_iterative_pruning_config,
    create_lottery_ticket_config,
    create_pruning_config,
    create_pruning_stats,
    estimate_speedup,
    format_pruning_stats,
    get_pruning_method,
    get_pruning_schedule,
    get_pruning_scope,
    get_recommended_pruning_config,
    list_pruning_methods,
    list_pruning_schedules,
    list_pruning_scopes,
    schedule_sparsity,
    validate_iterative_pruning_config,
    validate_lottery_ticket_config,
    validate_pruning_config,
    validate_pruning_stats,
)


class TestPruningMethod:
    """Tests for PruningMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in PruningMethod:
            assert isinstance(method.value, str)

    def test_magnitude_value(self) -> None:
        """Magnitude has correct value."""
        assert PruningMethod.MAGNITUDE.value == "magnitude"

    def test_movement_value(self) -> None:
        """Movement has correct value."""
        assert PruningMethod.MOVEMENT.value == "movement"

    def test_lottery_ticket_value(self) -> None:
        """Lottery ticket has correct value."""
        assert PruningMethod.LOTTERY_TICKET.value == "lottery_ticket"

    def test_structured_value(self) -> None:
        """Structured has correct value."""
        assert PruningMethod.STRUCTURED.value == "structured"

    def test_unstructured_value(self) -> None:
        """Unstructured has correct value."""
        assert PruningMethod.UNSTRUCTURED.value == "unstructured"

    def test_gradual_value(self) -> None:
        """Gradual has correct value."""
        assert PruningMethod.GRADUAL.value == "gradual"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_PRUNING_METHODS is a frozenset."""
        assert isinstance(VALID_PRUNING_METHODS, frozenset)
        assert len(VALID_PRUNING_METHODS) == 6


class TestPruningSchedule:
    """Tests for PruningSchedule enum."""

    def test_all_schedules_have_values(self) -> None:
        """All schedules have string values."""
        for schedule in PruningSchedule:
            assert isinstance(schedule.value, str)

    def test_one_shot_value(self) -> None:
        """One-shot has correct value."""
        assert PruningSchedule.ONE_SHOT.value == "one_shot"

    def test_iterative_value(self) -> None:
        """Iterative has correct value."""
        assert PruningSchedule.ITERATIVE.value == "iterative"

    def test_cubic_value(self) -> None:
        """Cubic has correct value."""
        assert PruningSchedule.CUBIC.value == "cubic"

    def test_linear_value(self) -> None:
        """Linear has correct value."""
        assert PruningSchedule.LINEAR.value == "linear"

    def test_valid_schedules_frozenset(self) -> None:
        """VALID_PRUNING_SCHEDULES is a frozenset."""
        assert isinstance(VALID_PRUNING_SCHEDULES, frozenset)
        assert len(VALID_PRUNING_SCHEDULES) == 4


class TestPruningScope:
    """Tests for PruningScope enum."""

    def test_all_scopes_have_values(self) -> None:
        """All scopes have string values."""
        for scope in PruningScope:
            assert isinstance(scope.value, str)

    def test_global_unstructured_value(self) -> None:
        """Global unstructured has correct value."""
        assert PruningScope.GLOBAL_UNSTRUCTURED.value == "global_unstructured"

    def test_local_unstructured_value(self) -> None:
        """Local unstructured has correct value."""
        assert PruningScope.LOCAL_UNSTRUCTURED.value == "local_unstructured"

    def test_structured_heads_value(self) -> None:
        """Structured heads has correct value."""
        assert PruningScope.STRUCTURED_HEADS.value == "structured_heads"

    def test_structured_neurons_value(self) -> None:
        """Structured neurons has correct value."""
        assert PruningScope.STRUCTURED_NEURONS.value == "structured_neurons"

    def test_valid_scopes_frozenset(self) -> None:
        """VALID_PRUNING_SCOPES is a frozenset."""
        assert isinstance(VALID_PRUNING_SCOPES, frozenset)
        assert len(VALID_PRUNING_SCOPES) == 4


class TestPruningConfig:
    """Tests for PruningConfig dataclass."""

    def test_create_config(self) -> None:
        """Create pruning config."""
        config = PruningConfig(
            method=PruningMethod.MAGNITUDE,
            target_sparsity=0.5,
            schedule=PruningSchedule.ONE_SHOT,
            scope=PruningScope.GLOBAL_UNSTRUCTURED,
        )
        assert config.method == PruningMethod.MAGNITUDE
        assert config.target_sparsity == pytest.approx(0.5)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            0.5,
            PruningSchedule.ONE_SHOT,
            PruningScope.GLOBAL_UNSTRUCTURED,
        )
        with pytest.raises(AttributeError):
            config.target_sparsity = 0.7  # type: ignore[misc]


class TestIterativePruningConfig:
    """Tests for IterativePruningConfig dataclass."""

    def test_create_config(self) -> None:
        """Create iterative pruning config."""
        config = IterativePruningConfig(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            pruning_steps=10,
            rewind_epoch=0,
        )
        assert config.initial_sparsity == pytest.approx(0.0)
        assert config.final_sparsity == pytest.approx(0.9)
        assert config.pruning_steps == 10

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = IterativePruningConfig(0.0, 0.9, 10, 0)
        with pytest.raises(AttributeError):
            config.final_sparsity = 0.95  # type: ignore[misc]


class TestLotteryTicketConfig:
    """Tests for LotteryTicketConfig dataclass."""

    def test_create_config(self) -> None:
        """Create lottery ticket config."""
        config = LotteryTicketConfig(
            rewind_epoch=0,
            num_iterations=15,
            target_sparsity=0.9,
        )
        assert config.rewind_epoch == 0
        assert config.num_iterations == 15
        assert config.target_sparsity == pytest.approx(0.9)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = LotteryTicketConfig(0, 15, 0.9)
        with pytest.raises(AttributeError):
            config.num_iterations = 20  # type: ignore[misc]


class TestPruningStats:
    """Tests for PruningStats dataclass."""

    def test_create_stats(self) -> None:
        """Create pruning stats."""
        stats = PruningStats(
            original_params=110_000_000,
            pruned_params=55_000_000,
            sparsity=0.5,
            speedup_factor=1.8,
        )
        assert stats.original_params == 110_000_000
        assert stats.sparsity == pytest.approx(0.5)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = PruningStats(110_000_000, 55_000_000, 0.5, 1.8)
        with pytest.raises(AttributeError):
            stats.sparsity = 0.6  # type: ignore[misc]


class TestValidatePruningConfig:
    """Tests for validate_pruning_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            0.5,
            PruningSchedule.ONE_SHOT,
            PruningScope.GLOBAL_UNSTRUCTURED,
        )
        validate_pruning_config(config)

    def test_sparsity_above_one_raises(self) -> None:
        """Sparsity > 1 raises ValueError."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            1.5,
            PruningSchedule.ONE_SHOT,
            PruningScope.GLOBAL_UNSTRUCTURED,
        )
        with pytest.raises(ValueError, match="target_sparsity must be between 0 and 1"):
            validate_pruning_config(config)

    def test_negative_sparsity_raises(self) -> None:
        """Negative sparsity raises ValueError."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            -0.1,
            PruningSchedule.ONE_SHOT,
            PruningScope.GLOBAL_UNSTRUCTURED,
        )
        with pytest.raises(ValueError, match="target_sparsity must be between 0 and 1"):
            validate_pruning_config(config)

    def test_edge_sparsity_zero(self) -> None:
        """Sparsity of 0 is valid."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            0.0,
            PruningSchedule.ONE_SHOT,
            PruningScope.GLOBAL_UNSTRUCTURED,
        )
        validate_pruning_config(config)

    def test_edge_sparsity_one(self) -> None:
        """Sparsity of 1 is valid."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            1.0,
            PruningSchedule.ONE_SHOT,
            PruningScope.GLOBAL_UNSTRUCTURED,
        )
        validate_pruning_config(config)


class TestValidateIterativePruningConfig:
    """Tests for validate_iterative_pruning_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = IterativePruningConfig(0.0, 0.9, 10, 0)
        validate_iterative_pruning_config(config)

    def test_final_less_than_initial_raises(self) -> None:
        """Final < initial sparsity raises ValueError."""
        config = IterativePruningConfig(0.5, 0.3, 10, 0)
        with pytest.raises(ValueError, match=r"final_sparsity.*must be >= initial"):
            validate_iterative_pruning_config(config)

    def test_negative_initial_sparsity_raises(self) -> None:
        """Negative initial sparsity raises ValueError."""
        config = IterativePruningConfig(-0.1, 0.9, 10, 0)
        with pytest.raises(ValueError, match="initial_sparsity must be between"):
            validate_iterative_pruning_config(config)

    def test_initial_above_one_raises(self) -> None:
        """Initial > 1 raises ValueError."""
        config = IterativePruningConfig(1.5, 0.9, 10, 0)
        with pytest.raises(ValueError, match="initial_sparsity must be between"):
            validate_iterative_pruning_config(config)

    def test_final_above_one_raises(self) -> None:
        """Final > 1 raises ValueError."""
        config = IterativePruningConfig(0.0, 1.5, 10, 0)
        with pytest.raises(ValueError, match="final_sparsity must be between"):
            validate_iterative_pruning_config(config)

    def test_zero_pruning_steps_raises(self) -> None:
        """Zero pruning steps raises ValueError."""
        config = IterativePruningConfig(0.0, 0.9, 0, 0)
        with pytest.raises(ValueError, match="pruning_steps must be positive"):
            validate_iterative_pruning_config(config)

    def test_negative_pruning_steps_raises(self) -> None:
        """Negative pruning steps raises ValueError."""
        config = IterativePruningConfig(0.0, 0.9, -1, 0)
        with pytest.raises(ValueError, match="pruning_steps must be positive"):
            validate_iterative_pruning_config(config)

    def test_negative_rewind_epoch_raises(self) -> None:
        """Negative rewind epoch raises ValueError."""
        config = IterativePruningConfig(0.0, 0.9, 10, -1)
        with pytest.raises(ValueError, match="rewind_epoch must be non-negative"):
            validate_iterative_pruning_config(config)


class TestValidateLotteryTicketConfig:
    """Tests for validate_lottery_ticket_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LotteryTicketConfig(0, 15, 0.9)
        validate_lottery_ticket_config(config)

    def test_negative_rewind_epoch_raises(self) -> None:
        """Negative rewind epoch raises ValueError."""
        config = LotteryTicketConfig(-1, 15, 0.9)
        with pytest.raises(ValueError, match="rewind_epoch must be non-negative"):
            validate_lottery_ticket_config(config)

    def test_zero_iterations_raises(self) -> None:
        """Zero iterations raises ValueError."""
        config = LotteryTicketConfig(0, 0, 0.9)
        with pytest.raises(ValueError, match="num_iterations must be positive"):
            validate_lottery_ticket_config(config)

    def test_negative_iterations_raises(self) -> None:
        """Negative iterations raises ValueError."""
        config = LotteryTicketConfig(0, -1, 0.9)
        with pytest.raises(ValueError, match="num_iterations must be positive"):
            validate_lottery_ticket_config(config)

    def test_sparsity_above_one_raises(self) -> None:
        """Sparsity > 1 raises ValueError."""
        config = LotteryTicketConfig(0, 15, 1.5)
        with pytest.raises(ValueError, match="target_sparsity must be between"):
            validate_lottery_ticket_config(config)

    def test_negative_sparsity_raises(self) -> None:
        """Negative sparsity raises ValueError."""
        config = LotteryTicketConfig(0, 15, -0.1)
        with pytest.raises(ValueError, match="target_sparsity must be between"):
            validate_lottery_ticket_config(config)


class TestValidatePruningStats:
    """Tests for validate_pruning_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats pass validation."""
        stats = PruningStats(110_000_000, 55_000_000, 0.5, 1.8)
        validate_pruning_stats(stats)

    def test_zero_original_params_raises(self) -> None:
        """Zero original params raises ValueError."""
        stats = PruningStats(0, 55_000_000, 0.5, 1.8)
        with pytest.raises(ValueError, match="original_params must be positive"):
            validate_pruning_stats(stats)

    def test_negative_original_params_raises(self) -> None:
        """Negative original params raises ValueError."""
        stats = PruningStats(-1, 55_000_000, 0.5, 1.8)
        with pytest.raises(ValueError, match="original_params must be positive"):
            validate_pruning_stats(stats)

    def test_negative_pruned_params_raises(self) -> None:
        """Negative pruned params raises ValueError."""
        stats = PruningStats(110_000_000, -1, 0.5, 1.8)
        with pytest.raises(ValueError, match="pruned_params must be non-negative"):
            validate_pruning_stats(stats)

    def test_pruned_exceeds_original_raises(self) -> None:
        """Pruned > original raises ValueError."""
        stats = PruningStats(50_000_000, 100_000_000, 0.5, 1.8)
        with pytest.raises(ValueError, match=r"pruned_params.*cannot exceed"):
            validate_pruning_stats(stats)

    def test_sparsity_above_one_raises(self) -> None:
        """Sparsity > 1 raises ValueError."""
        stats = PruningStats(110_000_000, 55_000_000, 1.5, 1.8)
        with pytest.raises(ValueError, match="sparsity must be between"):
            validate_pruning_stats(stats)

    def test_speedup_below_one_raises(self) -> None:
        """Speedup < 1 raises ValueError."""
        stats = PruningStats(110_000_000, 55_000_000, 0.5, 0.5)
        with pytest.raises(ValueError, match="speedup_factor must be >= 1"):
            validate_pruning_stats(stats)


class TestCreatePruningConfig:
    """Tests for create_pruning_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_pruning_config()
        assert config.method == PruningMethod.MAGNITUDE
        assert config.target_sparsity == pytest.approx(0.5)
        assert config.schedule == PruningSchedule.ONE_SHOT
        assert config.scope == PruningScope.GLOBAL_UNSTRUCTURED

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_pruning_config(
            method="movement",
            target_sparsity=0.7,
            schedule="iterative",
            scope="structured_heads",
        )
        assert config.method == PruningMethod.MOVEMENT
        assert config.target_sparsity == pytest.approx(0.7)
        assert config.schedule == PruningSchedule.ITERATIVE
        assert config.scope == PruningScope.STRUCTURED_HEADS

    def test_with_enum_values(self) -> None:
        """Create with enum values."""
        config = create_pruning_config(
            method=PruningMethod.LOTTERY_TICKET,
            schedule=PruningSchedule.CUBIC,
            scope=PruningScope.LOCAL_UNSTRUCTURED,
        )
        assert config.method == PruningMethod.LOTTERY_TICKET
        assert config.schedule == PruningSchedule.CUBIC
        assert config.scope == PruningScope.LOCAL_UNSTRUCTURED

    @pytest.mark.parametrize(
        "method",
        [
            "magnitude",
            "movement",
            "lottery_ticket",
            "structured",
            "unstructured",
            "gradual",
        ],
    )
    def test_all_methods(self, method: str) -> None:
        """Test all pruning methods."""
        config = create_pruning_config(method=method)
        assert config.method.value == method

    def test_invalid_sparsity_raises(self) -> None:
        """Invalid sparsity raises ValueError."""
        with pytest.raises(ValueError, match="target_sparsity must be between"):
            create_pruning_config(target_sparsity=1.5)


class TestCreateIterativePruningConfig:
    """Tests for create_iterative_pruning_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_iterative_pruning_config()
        assert config.initial_sparsity == pytest.approx(0.0)
        assert config.final_sparsity == pytest.approx(0.9)
        assert config.pruning_steps == 10
        assert config.rewind_epoch == 0

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_iterative_pruning_config(
            initial_sparsity=0.3,
            final_sparsity=0.95,
            pruning_steps=20,
            rewind_epoch=1,
        )
        assert config.initial_sparsity == pytest.approx(0.3)
        assert config.final_sparsity == pytest.approx(0.95)
        assert config.pruning_steps == 20
        assert config.rewind_epoch == 1

    def test_invalid_steps_raises(self) -> None:
        """Invalid pruning steps raises ValueError."""
        with pytest.raises(ValueError, match="pruning_steps must be positive"):
            create_iterative_pruning_config(pruning_steps=0)

    def test_final_less_than_initial_raises(self) -> None:
        """Final < initial raises ValueError."""
        with pytest.raises(ValueError, match=r"final_sparsity.*must be >= initial"):
            create_iterative_pruning_config(initial_sparsity=0.8, final_sparsity=0.5)


class TestCreateLotteryTicketConfig:
    """Tests for create_lottery_ticket_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_lottery_ticket_config()
        assert config.rewind_epoch == 0
        assert config.num_iterations == 15
        assert config.target_sparsity == pytest.approx(0.9)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_lottery_ticket_config(
            rewind_epoch=1,
            num_iterations=20,
            target_sparsity=0.95,
        )
        assert config.rewind_epoch == 1
        assert config.num_iterations == 20
        assert config.target_sparsity == pytest.approx(0.95)

    def test_invalid_iterations_raises(self) -> None:
        """Invalid iterations raises ValueError."""
        with pytest.raises(ValueError, match="num_iterations must be positive"):
            create_lottery_ticket_config(num_iterations=0)

    def test_negative_rewind_raises(self) -> None:
        """Negative rewind epoch raises ValueError."""
        with pytest.raises(ValueError, match="rewind_epoch must be non-negative"):
            create_lottery_ticket_config(rewind_epoch=-1)


class TestCreatePruningStats:
    """Tests for create_pruning_stats function."""

    def test_auto_calculate_sparsity(self) -> None:
        """Auto-calculate sparsity."""
        stats = create_pruning_stats(100, 50)
        assert stats.sparsity == pytest.approx(0.5)

    def test_auto_calculate_speedup(self) -> None:
        """Auto-calculate speedup."""
        stats = create_pruning_stats(100, 50)
        assert stats.speedup_factor > 1.0

    def test_explicit_values(self) -> None:
        """Use explicit values."""
        stats = create_pruning_stats(
            original_params=100_000,
            pruned_params=10_000,
            sparsity=0.9,
            speedup_factor=2.5,
        )
        assert stats.sparsity == pytest.approx(0.9)
        assert stats.speedup_factor == pytest.approx(2.5)

    def test_zero_original_raises(self) -> None:
        """Zero original params raises ValueError."""
        with pytest.raises(ValueError, match="original_params must be positive"):
            create_pruning_stats(0, 50_000)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_pruning_methods_sorted(self) -> None:
        """Returns sorted list."""
        methods = list_pruning_methods()
        assert methods == sorted(methods)
        assert "magnitude" in methods
        assert "lottery_ticket" in methods

    def test_list_pruning_schedules_sorted(self) -> None:
        """Returns sorted list."""
        schedules = list_pruning_schedules()
        assert schedules == sorted(schedules)
        assert "one_shot" in schedules
        assert "iterative" in schedules

    def test_list_pruning_scopes_sorted(self) -> None:
        """Returns sorted list."""
        scopes = list_pruning_scopes()
        assert scopes == sorted(scopes)
        assert "global_unstructured" in scopes
        assert "structured_heads" in scopes


class TestGetPruningMethod:
    """Tests for get_pruning_method function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("magnitude", PruningMethod.MAGNITUDE),
            ("movement", PruningMethod.MOVEMENT),
            ("lottery_ticket", PruningMethod.LOTTERY_TICKET),
            ("structured", PruningMethod.STRUCTURED),
            ("unstructured", PruningMethod.UNSTRUCTURED),
            ("gradual", PruningMethod.GRADUAL),
        ],
    )
    def test_all_methods(self, name: str, expected: PruningMethod) -> None:
        """Test all valid methods."""
        assert get_pruning_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_pruning_method("invalid")


class TestGetPruningSchedule:
    """Tests for get_pruning_schedule function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("one_shot", PruningSchedule.ONE_SHOT),
            ("iterative", PruningSchedule.ITERATIVE),
            ("cubic", PruningSchedule.CUBIC),
            ("linear", PruningSchedule.LINEAR),
        ],
    )
    def test_all_schedules(self, name: str, expected: PruningSchedule) -> None:
        """Test all valid schedules."""
        assert get_pruning_schedule(name) == expected

    def test_invalid_schedule_raises(self) -> None:
        """Invalid schedule raises ValueError."""
        with pytest.raises(ValueError, match="schedule must be one of"):
            get_pruning_schedule("invalid")


class TestGetPruningScope:
    """Tests for get_pruning_scope function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("global_unstructured", PruningScope.GLOBAL_UNSTRUCTURED),
            ("local_unstructured", PruningScope.LOCAL_UNSTRUCTURED),
            ("structured_heads", PruningScope.STRUCTURED_HEADS),
            ("structured_neurons", PruningScope.STRUCTURED_NEURONS),
        ],
    )
    def test_all_scopes(self, name: str, expected: PruningScope) -> None:
        """Test all valid scopes."""
        assert get_pruning_scope(name) == expected

    def test_invalid_scope_raises(self) -> None:
        """Invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="scope must be one of"):
            get_pruning_scope("invalid")


class TestCalculateSparsity:
    """Tests for calculate_sparsity function."""

    def test_half_sparsity(self) -> None:
        """Calculate 50% sparsity."""
        assert calculate_sparsity(100, 50) == pytest.approx(0.5)

    def test_high_sparsity(self) -> None:
        """Calculate 90% sparsity."""
        assert calculate_sparsity(1000, 100) == pytest.approx(0.9)

    def test_no_sparsity(self) -> None:
        """Calculate 0% sparsity."""
        assert calculate_sparsity(100, 100) == pytest.approx(0.0)

    def test_full_sparsity(self) -> None:
        """Calculate 100% sparsity."""
        assert calculate_sparsity(100, 0) == pytest.approx(1.0)

    def test_zero_original_raises(self) -> None:
        """Zero original params raises ValueError."""
        with pytest.raises(ValueError, match="original_params must be positive"):
            calculate_sparsity(0, 50)

    def test_negative_original_raises(self) -> None:
        """Negative original params raises ValueError."""
        with pytest.raises(ValueError, match="original_params must be positive"):
            calculate_sparsity(-100, 50)

    def test_negative_remaining_raises(self) -> None:
        """Negative remaining params raises ValueError."""
        with pytest.raises(ValueError, match="remaining_params must be non-negative"):
            calculate_sparsity(100, -50)

    def test_remaining_exceeds_original_raises(self) -> None:
        """Remaining > original raises ValueError."""
        with pytest.raises(ValueError, match=r"remaining_params.*cannot exceed"):
            calculate_sparsity(50, 100)

    @given(
        original=st.integers(min_value=1, max_value=1_000_000),
        fraction=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_property_sparsity_range(self, original: int, fraction: float) -> None:
        """Sparsity is always between 0 and 1."""
        remaining = int(original * fraction)
        sparsity = calculate_sparsity(original, remaining)
        assert 0 <= sparsity <= 1


class TestEstimateSpeedup:
    """Tests for estimate_speedup function."""

    def test_no_sparsity_no_speedup(self) -> None:
        """Zero sparsity gives speedup of 1."""
        assert estimate_speedup(0.0) == pytest.approx(1.0)

    def test_half_sparsity_speedup(self) -> None:
        """50% sparsity gives speedup > 1."""
        speedup = estimate_speedup(0.5)
        assert 1.0 < speedup < 2.0

    def test_high_sparsity_higher_speedup(self) -> None:
        """Higher sparsity gives higher speedup."""
        assert estimate_speedup(0.9) > estimate_speedup(0.5)

    def test_sparsity_above_one_raises(self) -> None:
        """Sparsity > 1 raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            estimate_speedup(1.5)

    def test_negative_sparsity_raises(self) -> None:
        """Negative sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            estimate_speedup(-0.1)

    def test_invalid_efficiency_raises(self) -> None:
        """Invalid efficiency factor raises ValueError."""
        with pytest.raises(ValueError, match="efficiency_factor must be between"):
            estimate_speedup(0.5, efficiency_factor=0)

    def test_efficiency_above_one_raises(self) -> None:
        """Efficiency > 1 raises ValueError."""
        with pytest.raises(ValueError, match="efficiency_factor must be between"):
            estimate_speedup(0.5, efficiency_factor=1.5)

    def test_custom_efficiency(self) -> None:
        """Custom efficiency factor affects speedup."""
        speedup_high = estimate_speedup(0.5, efficiency_factor=0.9)
        speedup_low = estimate_speedup(0.5, efficiency_factor=0.3)
        assert speedup_high > speedup_low

    @given(sparsity=st.floats(min_value=0.0, max_value=0.99))
    def test_property_speedup_at_least_one(self, sparsity: float) -> None:
        """Speedup is always >= 1."""
        assert estimate_speedup(sparsity) >= 1.0


class TestCalculatePruningMask:
    """Tests for calculate_pruning_mask function."""

    def test_basic_mask(self) -> None:
        """Calculate basic pruning mask."""
        weights = (0.1, 0.5, 0.2, 0.8, 0.3)
        mask = calculate_pruning_mask(weights, 0.4)
        assert sum(mask) == 3  # 60% kept
        assert mask == (False, True, False, True, True)

    def test_empty_weights(self) -> None:
        """Empty weights returns empty mask."""
        assert calculate_pruning_mask((), 0.5) == ()

    def test_zero_sparsity(self) -> None:
        """Zero sparsity keeps all weights."""
        weights = (0.1, 0.5, 0.2, 0.8, 0.3)
        mask = calculate_pruning_mask(weights, 0.0)
        assert all(mask)

    def test_full_sparsity(self) -> None:
        """Full sparsity prunes all weights."""
        weights = (0.1, 0.5, 0.2, 0.8, 0.3)
        mask = calculate_pruning_mask(weights, 1.0)
        assert not any(mask)

    def test_sparsity_above_one_raises(self) -> None:
        """Sparsity > 1 raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            calculate_pruning_mask((0.1, 0.2), 1.5)

    def test_negative_sparsity_raises(self) -> None:
        """Negative sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            calculate_pruning_mask((0.1, 0.2), -0.1)

    def test_magnitude_pruning_keeps_largest(self) -> None:
        """Magnitude pruning keeps largest weights."""
        weights = (0.1, 0.9, 0.2, 0.8, 0.3)
        mask = calculate_pruning_mask(weights, 0.6, PruningMethod.MAGNITUDE)
        # Should keep 0.9 and 0.8
        assert mask[1] is True  # 0.9
        assert mask[3] is True  # 0.8

    def test_negative_weights_handled(self) -> None:
        """Negative weights use absolute value."""
        weights = (-0.9, 0.1, -0.8, 0.2, 0.3)
        mask = calculate_pruning_mask(weights, 0.6, PruningMethod.MAGNITUDE)
        # Should keep -0.9 and -0.8 (largest absolute values)
        assert mask[0] is True  # -0.9
        assert mask[2] is True  # -0.8


class TestScheduleSparsity:
    """Tests for schedule_sparsity function."""

    def test_linear_start(self) -> None:
        """Linear schedule at start."""
        sparsity = schedule_sparsity(0, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        assert sparsity == pytest.approx(0.0)

    def test_linear_middle(self) -> None:
        """Linear schedule at middle."""
        sparsity = schedule_sparsity(500, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        assert sparsity == pytest.approx(0.45)

    def test_linear_end(self) -> None:
        """Linear schedule at end."""
        sparsity = schedule_sparsity(1000, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        assert sparsity == pytest.approx(0.9)

    def test_one_shot_before_end(self) -> None:
        """One-shot returns initial before end."""
        sparsity = schedule_sparsity(500, 1000, 0.0, 0.9, PruningSchedule.ONE_SHOT)
        assert sparsity == pytest.approx(0.0)

    def test_one_shot_at_end(self) -> None:
        """One-shot returns final at end."""
        sparsity = schedule_sparsity(1000, 1000, 0.0, 0.9, PruningSchedule.ONE_SHOT)
        assert sparsity == pytest.approx(0.9)

    def test_cubic_slower_start(self) -> None:
        """Cubic has slower start than linear."""
        linear = schedule_sparsity(500, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        cubic = schedule_sparsity(500, 1000, 0.0, 0.9, PruningSchedule.CUBIC)
        assert cubic < linear

    def test_iterative_same_as_linear(self) -> None:
        """Iterative behaves like linear for progress."""
        linear = schedule_sparsity(500, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        iterative = schedule_sparsity(500, 1000, 0.0, 0.9, PruningSchedule.ITERATIVE)
        assert iterative == pytest.approx(linear)

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        with pytest.raises(ValueError, match="current_step must be non-negative"):
            schedule_sparsity(-1, 1000, 0.0, 0.9, PruningSchedule.LINEAR)

    def test_zero_total_steps_raises(self) -> None:
        """Zero total steps raises ValueError."""
        with pytest.raises(ValueError, match="total_steps must be positive"):
            schedule_sparsity(0, 0, 0.0, 0.9, PruningSchedule.LINEAR)

    def test_step_exceeds_total_raises(self) -> None:
        """Step > total raises ValueError."""
        with pytest.raises(ValueError, match=r"current_step.*cannot exceed"):
            schedule_sparsity(1001, 1000, 0.0, 0.9, PruningSchedule.LINEAR)

    def test_invalid_initial_sparsity_raises(self) -> None:
        """Invalid initial sparsity raises ValueError."""
        with pytest.raises(ValueError, match="initial_sparsity must be between"):
            schedule_sparsity(500, 1000, -0.1, 0.9, PruningSchedule.LINEAR)

    def test_invalid_final_sparsity_raises(self) -> None:
        """Invalid final sparsity raises ValueError."""
        with pytest.raises(ValueError, match="final_sparsity must be between"):
            schedule_sparsity(500, 1000, 0.0, 1.5, PruningSchedule.LINEAR)

    @given(
        step=st.integers(min_value=0, max_value=1000),
        initial=st.floats(min_value=0.0, max_value=0.5),
        final=st.floats(min_value=0.5, max_value=1.0),
    )
    def test_property_sparsity_in_range(
        self, step: int, initial: float, final: float
    ) -> None:
        """Sparsity is always between initial and final."""
        sparsity = schedule_sparsity(step, 1000, initial, final, PruningSchedule.LINEAR)
        assert initial <= sparsity <= final


class TestFormatPruningStats:
    """Tests for format_pruning_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_pruning_stats(110_000_000, 55_000_000)
        formatted = format_pruning_stats(stats)
        assert "Sparsity: 50.0%" in formatted
        assert "Original: 110.00M" in formatted

    def test_billion_params(self) -> None:
        """Format billion parameter model."""
        stats = create_pruning_stats(7_000_000_000, 3_500_000_000)
        formatted = format_pruning_stats(stats)
        assert "Original: 7.00B" in formatted

    def test_thousand_params(self) -> None:
        """Format thousand parameter model."""
        stats = create_pruning_stats(50_000, 25_000)
        formatted = format_pruning_stats(stats)
        assert "Original: 50.00K" in formatted

    def test_small_params(self) -> None:
        """Format small parameter count."""
        stats = create_pruning_stats(500, 250)
        formatted = format_pruning_stats(stats)
        assert "Original: 500" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_pruning_stats(100_000, 50_000)
        formatted = format_pruning_stats(stats)
        assert "Original:" in formatted
        assert "Remaining:" in formatted
        assert "Sparsity:" in formatted
        assert "Speedup:" in formatted


class TestGetRecommendedPruningConfig:
    """Tests for get_recommended_pruning_config function."""

    def test_default_config(self) -> None:
        """Get default recommended config."""
        config = get_recommended_pruning_config()
        assert config.target_sparsity == pytest.approx(0.5)
        assert config.method == PruningMethod.MAGNITUDE

    def test_small_model_classification(self) -> None:
        """Small model for classification."""
        config = get_recommended_pruning_config("small", "classification")
        assert config.target_sparsity == pytest.approx(0.3)
        assert config.schedule == PruningSchedule.ONE_SHOT

    def test_large_model_generation(self) -> None:
        """Large model for generation."""
        config = get_recommended_pruning_config("large", "generation")
        assert config.target_sparsity == pytest.approx(0.7)
        assert config.method == PruningMethod.GRADUAL
        assert config.schedule == PruningSchedule.ITERATIVE

    def test_xl_model_embedding(self) -> None:
        """XL model for embedding."""
        config = get_recommended_pruning_config("xl", "embedding")
        assert config.target_sparsity == pytest.approx(0.8)
        assert config.method == PruningMethod.STRUCTURED
        assert config.scope == PruningScope.STRUCTURED_NEURONS

    @pytest.mark.parametrize("size", ["small", "base", "large", "xl"])
    def test_all_sizes(self, size: str) -> None:
        """Test all model sizes."""
        config = get_recommended_pruning_config(size)
        assert config is not None

    @pytest.mark.parametrize("task", ["classification", "generation", "embedding"])
    def test_all_tasks(self, task: str) -> None:
        """Test all task types."""
        config = get_recommended_pruning_config(task=task)
        assert config is not None

    def test_invalid_size_raises(self) -> None:
        """Invalid model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_pruning_config("invalid")

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_pruning_config(task="invalid")
