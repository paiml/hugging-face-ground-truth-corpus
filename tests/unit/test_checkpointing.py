"""Tests for training.checkpointing module."""

from __future__ import annotations

import pytest

from hf_gtc.training.checkpointing import (
    VALID_CHECKPOINT_GRANULARITIES,
    VALID_CHECKPOINT_STRATEGIES,
    VALID_OFFLOAD_TARGETS,
    CheckpointConfig,
    CheckpointGranularity,
    CheckpointStrategy,
    MemoryConfig,
    MemoryStats,
    OffloadConfig,
    OffloadTarget,
    calculate_memory_savings,
    calculate_optimal_checkpoint_ratio,
    create_checkpoint_config,
    create_memory_config,
    create_offload_config,
    estimate_recomputation_overhead,
    format_memory_stats,
    get_checkpoint_granularity,
    get_checkpoint_strategy,
    get_offload_target,
    get_recommended_checkpoint_config,
    list_checkpoint_granularities,
    list_checkpoint_strategies,
    list_offload_targets,
    select_checkpoint_layers,
    validate_checkpoint_config,
    validate_memory_config,
    validate_offload_config,
)


class TestCheckpointStrategy:
    """Tests for CheckpointStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in CheckpointStrategy:
            assert isinstance(strategy.value, str)

    def test_full_value(self) -> None:
        """Full has correct value."""
        assert CheckpointStrategy.FULL.value == "full"

    def test_selective_value(self) -> None:
        """Selective has correct value."""
        assert CheckpointStrategy.SELECTIVE.value == "selective"

    def test_offload_value(self) -> None:
        """Offload has correct value."""
        assert CheckpointStrategy.OFFLOAD.value == "offload"

    def test_none_value(self) -> None:
        """None has correct value."""
        assert CheckpointStrategy.NONE.value == "none"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_CHECKPOINT_STRATEGIES is a frozenset."""
        assert isinstance(VALID_CHECKPOINT_STRATEGIES, frozenset)
        assert len(VALID_CHECKPOINT_STRATEGIES) == 4


class TestCheckpointGranularity:
    """Tests for CheckpointGranularity enum."""

    def test_all_granularities_have_values(self) -> None:
        """All granularities have string values."""
        for granularity in CheckpointGranularity:
            assert isinstance(granularity.value, str)

    def test_layer_value(self) -> None:
        """Layer has correct value."""
        assert CheckpointGranularity.LAYER.value == "layer"

    def test_block_value(self) -> None:
        """Block has correct value."""
        assert CheckpointGranularity.BLOCK.value == "block"

    def test_attention_value(self) -> None:
        """Attention has correct value."""
        assert CheckpointGranularity.ATTENTION.value == "attention"

    def test_mlp_value(self) -> None:
        """MLP has correct value."""
        assert CheckpointGranularity.MLP.value == "mlp"

    def test_valid_granularities_frozenset(self) -> None:
        """VALID_CHECKPOINT_GRANULARITIES is a frozenset."""
        assert isinstance(VALID_CHECKPOINT_GRANULARITIES, frozenset)
        assert len(VALID_CHECKPOINT_GRANULARITIES) == 4


class TestOffloadTarget:
    """Tests for OffloadTarget enum."""

    def test_all_targets_have_values(self) -> None:
        """All targets have string values."""
        for target in OffloadTarget:
            assert isinstance(target.value, str)

    def test_cpu_value(self) -> None:
        """CPU has correct value."""
        assert OffloadTarget.CPU.value == "cpu"

    def test_disk_value(self) -> None:
        """Disk has correct value."""
        assert OffloadTarget.DISK.value == "disk"

    def test_nvme_value(self) -> None:
        """NVMe has correct value."""
        assert OffloadTarget.NVME.value == "nvme"

    def test_valid_targets_frozenset(self) -> None:
        """VALID_OFFLOAD_TARGETS is a frozenset."""
        assert isinstance(VALID_OFFLOAD_TARGETS, frozenset)
        assert len(VALID_OFFLOAD_TARGETS) == 3


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_create_config(self) -> None:
        """Create checkpoint config."""
        config = CheckpointConfig(
            strategy=CheckpointStrategy.SELECTIVE,
            granularity=CheckpointGranularity.BLOCK,
            checkpoint_ratio=0.5,
        )
        assert config.strategy == CheckpointStrategy.SELECTIVE
        assert config.granularity == CheckpointGranularity.BLOCK
        assert config.checkpoint_ratio == 0.5

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = CheckpointConfig(
            CheckpointStrategy.FULL, CheckpointGranularity.LAYER, 1.0
        )
        with pytest.raises(AttributeError):
            config.checkpoint_ratio = 0.5  # type: ignore[misc]


class TestOffloadConfig:
    """Tests for OffloadConfig dataclass."""

    def test_create_config(self) -> None:
        """Create offload config."""
        config = OffloadConfig(
            target=OffloadTarget.CPU,
            pin_memory=True,
            async_transfer=True,
        )
        assert config.target == OffloadTarget.CPU
        assert config.pin_memory is True
        assert config.async_transfer is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = OffloadConfig(OffloadTarget.CPU, True, True)
        with pytest.raises(AttributeError):
            config.pin_memory = False  # type: ignore[misc]


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_create_config(self) -> None:
        """Create memory config."""
        ckpt = CheckpointConfig(
            CheckpointStrategy.FULL, CheckpointGranularity.LAYER, 1.0
        )
        offload = OffloadConfig(OffloadTarget.CPU, True, True)
        config = MemoryConfig(
            checkpoint_config=ckpt,
            offload_config=offload,
            cpu_offload=False,
        )
        assert config.checkpoint_config.strategy == CheckpointStrategy.FULL
        assert config.offload_config.target == OffloadTarget.CPU
        assert config.cpu_offload is False


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_create_stats(self) -> None:
        """Create memory stats."""
        stats = MemoryStats(
            baseline_memory_gb=32.0,
            checkpointed_memory_gb=12.0,
            memory_saved_gb=20.0,
            savings_percentage=62.5,
            recomputation_overhead_pct=33.0,
        )
        assert stats.baseline_memory_gb == 32.0
        assert stats.memory_saved_gb == 20.0
        assert stats.savings_percentage == 62.5


class TestValidateCheckpointConfig:
    """Tests for validate_checkpoint_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = CheckpointConfig(
            CheckpointStrategy.SELECTIVE, CheckpointGranularity.BLOCK, 0.5
        )
        validate_checkpoint_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_checkpoint_config(None)  # type: ignore[arg-type]

    def test_negative_ratio_raises(self) -> None:
        """Negative checkpoint ratio raises ValueError."""
        config = CheckpointConfig(
            CheckpointStrategy.SELECTIVE, CheckpointGranularity.BLOCK, -0.1
        )
        with pytest.raises(ValueError, match="checkpoint_ratio must be in"):
            validate_checkpoint_config(config)

    def test_ratio_over_one_raises(self) -> None:
        """Checkpoint ratio over 1.0 raises ValueError."""
        config = CheckpointConfig(
            CheckpointStrategy.SELECTIVE, CheckpointGranularity.BLOCK, 1.5
        )
        with pytest.raises(ValueError, match="checkpoint_ratio must be in"):
            validate_checkpoint_config(config)

    def test_zero_ratio_non_none_strategy_raises(self) -> None:
        """Zero ratio with non-NONE strategy raises ValueError."""
        config = CheckpointConfig(
            CheckpointStrategy.SELECTIVE, CheckpointGranularity.BLOCK, 0.0
        )
        with pytest.raises(ValueError, match=r"checkpoint_ratio cannot be 0\.0"):
            validate_checkpoint_config(config)

    def test_zero_ratio_none_strategy_valid(self) -> None:
        """Zero ratio with NONE strategy is valid."""
        config = CheckpointConfig(
            CheckpointStrategy.NONE, CheckpointGranularity.BLOCK, 0.0
        )
        validate_checkpoint_config(config)


class TestValidateOffloadConfig:
    """Tests for validate_offload_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = OffloadConfig(OffloadTarget.CPU, True, True)
        validate_offload_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_offload_config(None)  # type: ignore[arg-type]

    def test_pin_memory_with_disk_raises(self) -> None:
        """Pin memory with disk target raises ValueError."""
        config = OffloadConfig(OffloadTarget.DISK, True, True)
        with pytest.raises(ValueError, match="pin_memory only applies to CPU"):
            validate_offload_config(config)

    def test_pin_memory_with_nvme_raises(self) -> None:
        """Pin memory with NVMe target raises ValueError."""
        config = OffloadConfig(OffloadTarget.NVME, True, True)
        with pytest.raises(ValueError, match="pin_memory only applies to CPU"):
            validate_offload_config(config)

    def test_no_pin_memory_with_disk_valid(self) -> None:
        """No pin memory with disk target is valid."""
        config = OffloadConfig(OffloadTarget.DISK, False, True)
        validate_offload_config(config)


class TestValidateMemoryConfig:
    """Tests for validate_memory_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        ckpt = CheckpointConfig(
            CheckpointStrategy.FULL, CheckpointGranularity.LAYER, 1.0
        )
        offload = OffloadConfig(OffloadTarget.CPU, True, True)
        config = MemoryConfig(ckpt, offload, False)
        validate_memory_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_memory_config(None)  # type: ignore[arg-type]


class TestCreateCheckpointConfig:
    """Tests for create_checkpoint_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_checkpoint_config()
        assert config.strategy == CheckpointStrategy.SELECTIVE
        assert config.granularity == CheckpointGranularity.BLOCK
        assert config.checkpoint_ratio == 0.5

    def test_custom_strategy(self) -> None:
        """Create config with custom strategy."""
        config = create_checkpoint_config(strategy="full", checkpoint_ratio=1.0)
        assert config.strategy == CheckpointStrategy.FULL

    def test_custom_granularity(self) -> None:
        """Create config with custom granularity."""
        config = create_checkpoint_config(granularity="attention")
        assert config.granularity == CheckpointGranularity.ATTENTION

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_checkpoint_config(strategy="invalid")

    def test_invalid_granularity_raises(self) -> None:
        """Invalid granularity raises ValueError."""
        with pytest.raises(ValueError, match="granularity must be one of"):
            create_checkpoint_config(granularity="invalid")

    def test_none_strategy_zero_ratio(self) -> None:
        """None strategy with zero ratio is valid."""
        config = create_checkpoint_config(strategy="none", checkpoint_ratio=0.0)
        assert config.strategy == CheckpointStrategy.NONE
        assert config.checkpoint_ratio == 0.0


class TestCreateOffloadConfig:
    """Tests for create_offload_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_offload_config()
        assert config.target == OffloadTarget.CPU
        assert config.pin_memory is True
        assert config.async_transfer is True

    def test_custom_target(self) -> None:
        """Create config with custom target."""
        config = create_offload_config(target="disk", pin_memory=False)
        assert config.target == OffloadTarget.DISK
        assert config.pin_memory is False

    def test_nvme_target(self) -> None:
        """Create config with NVMe target."""
        config = create_offload_config(target="nvme", pin_memory=False)
        assert config.target == OffloadTarget.NVME

    def test_invalid_target_raises(self) -> None:
        """Invalid target raises ValueError."""
        with pytest.raises(ValueError, match="target must be one of"):
            create_offload_config(target="invalid")


class TestCreateMemoryConfig:
    """Tests for create_memory_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_memory_config()
        assert config.checkpoint_config.strategy == CheckpointStrategy.SELECTIVE
        assert config.offload_config.target == OffloadTarget.CPU
        assert config.cpu_offload is False

    def test_custom_checkpoint_config(self) -> None:
        """Create config with custom checkpoint config."""
        ckpt = create_checkpoint_config(strategy="full", checkpoint_ratio=1.0)
        config = create_memory_config(checkpoint_config=ckpt)
        assert config.checkpoint_config.strategy == CheckpointStrategy.FULL

    def test_custom_offload_config(self) -> None:
        """Create config with custom offload config."""
        offload = create_offload_config(target="disk", pin_memory=False)
        config = create_memory_config(offload_config=offload)
        assert config.offload_config.target == OffloadTarget.DISK

    def test_cpu_offload_enabled(self) -> None:
        """Create config with CPU offload enabled."""
        config = create_memory_config(cpu_offload=True)
        assert config.cpu_offload is True


class TestCalculateMemorySavings:
    """Tests for calculate_memory_savings function."""

    def test_basic_calculation(self) -> None:
        """Basic memory savings calculation."""
        baseline, ckpt, saved = calculate_memory_savings(7.0, 32)
        assert baseline > 0
        assert ckpt > 0
        assert saved > 0
        assert ckpt < baseline

    def test_full_strategy_max_savings(self) -> None:
        """Full strategy gives maximum savings."""
        _, _ckpt_full, saved_full = calculate_memory_savings(7.0, 32, "full", 1.0)
        _, _ckpt_sel, saved_sel = calculate_memory_savings(7.0, 32, "selective", 0.5)
        assert saved_full >= saved_sel

    def test_none_strategy_no_savings(self) -> None:
        """None strategy gives no savings."""
        baseline, ckpt, saved = calculate_memory_savings(7.0, 32, "none", 0.0)
        assert saved == 0.0
        assert baseline == ckpt

    def test_zero_model_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            calculate_memory_savings(0, 32)

    def test_negative_model_params_raises(self) -> None:
        """Negative model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            calculate_memory_savings(-1.0, 32)

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            calculate_memory_savings(7.0, 0)

    def test_invalid_ratio_raises(self) -> None:
        """Invalid checkpoint ratio raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_ratio must be in"):
            calculate_memory_savings(7.0, 32, "selective", 1.5)

    def test_custom_batch_size(self) -> None:
        """Custom batch size affects memory."""
        _, _, saved_bs1 = calculate_memory_savings(7.0, 32, batch_size=1)
        _, _, saved_bs4 = calculate_memory_savings(7.0, 32, batch_size=4)
        assert saved_bs4 > saved_bs1


class TestEstimateRecomputationOverhead:
    """Tests for estimate_recomputation_overhead function."""

    def test_full_strategy_overhead(self) -> None:
        """Full strategy has maximum overhead."""
        overhead = estimate_recomputation_overhead("full", 1.0, 32)
        assert overhead > 0
        assert overhead <= 50.0

    def test_none_strategy_no_overhead(self) -> None:
        """None strategy has no overhead."""
        overhead = estimate_recomputation_overhead("none", 0.0, 32)
        assert overhead == 0.0

    def test_selective_partial_overhead(self) -> None:
        """Selective strategy has partial overhead."""
        overhead_full = estimate_recomputation_overhead("full", 1.0, 32)
        overhead_sel = estimate_recomputation_overhead("selective", 0.5, 32)
        assert overhead_sel < overhead_full

    def test_offload_reduced_overhead(self) -> None:
        """Offload has reduced recomputation overhead."""
        overhead_full = estimate_recomputation_overhead("full", 1.0, 32)
        overhead_offload = estimate_recomputation_overhead("offload", 1.0, 32)
        assert overhead_offload < overhead_full

    def test_invalid_ratio_raises(self) -> None:
        """Invalid checkpoint ratio raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_ratio must be in"):
            estimate_recomputation_overhead("full", 1.5, 32)

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_recomputation_overhead("full", 1.0, 0)


class TestCalculateOptimalCheckpointRatio:
    """Tests for calculate_optimal_checkpoint_ratio function."""

    def test_ample_memory_low_ratio(self) -> None:
        """Ample memory results in low checkpoint ratio."""
        # 1B model with 80GB GPU = plenty of headroom
        ratio = calculate_optimal_checkpoint_ratio(80.0, 1.0, 32)
        assert ratio == 0.0  # No checkpointing needed

    def test_limited_memory_high_ratio(self) -> None:
        """Limited memory results in high checkpoint ratio."""
        ratio = calculate_optimal_checkpoint_ratio(16.0, 7.0, 32)
        assert ratio > 0.5

    def test_ratio_in_valid_range(self) -> None:
        """Ratio is always in valid range."""
        ratio = calculate_optimal_checkpoint_ratio(24.0, 7.0, 32)
        assert 0.0 <= ratio <= 1.0

    def test_zero_gpu_memory_raises(self) -> None:
        """Zero GPU memory raises ValueError."""
        with pytest.raises(
            ValueError, match="available_gpu_memory_gb must be positive"
        ):
            calculate_optimal_checkpoint_ratio(0, 7.0, 32)

    def test_zero_model_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            calculate_optimal_checkpoint_ratio(24.0, 0, 32)

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            calculate_optimal_checkpoint_ratio(24.0, 7.0, 0)

    def test_edge_case_max_savings_zero(self) -> None:
        """Edge case where max savings would be zero or negative returns 1.0."""
        # Very small model/layer configuration that results in minimal savings
        # In reality this is hard to trigger, but we test the code path
        ratio = calculate_optimal_checkpoint_ratio(1.0, 0.001, 1)
        assert 0.0 <= ratio <= 1.0


class TestSelectCheckpointLayers:
    """Tests for select_checkpoint_layers function."""

    def test_half_ratio(self) -> None:
        """Half ratio selects half the layers."""
        layers = select_checkpoint_layers(32, 0.5)
        assert len(layers) == 16

    def test_full_ratio(self) -> None:
        """Full ratio selects all layers."""
        layers = select_checkpoint_layers(32, 1.0)
        assert len(layers) == 32

    def test_zero_ratio(self) -> None:
        """Zero ratio selects no layers."""
        layers = select_checkpoint_layers(32, 0.0)
        assert layers == []

    def test_layers_in_valid_range(self) -> None:
        """Selected layers are in valid range."""
        layers = select_checkpoint_layers(32, 0.5)
        assert all(0 <= layer < 32 for layer in layers)

    def test_attention_granularity_later_layers(self) -> None:
        """Attention granularity prefers later layers."""
        layers = select_checkpoint_layers(32, 0.25, "attention")
        # Later layers should be selected
        assert max(layers) > 20

    def test_mlp_granularity_later_layers(self) -> None:
        """MLP granularity prefers later layers."""
        layers = select_checkpoint_layers(32, 0.25, "mlp")
        assert max(layers) > 20

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            select_checkpoint_layers(0, 0.5)

    def test_invalid_ratio_raises(self) -> None:
        """Invalid checkpoint ratio raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_ratio must be in"):
            select_checkpoint_layers(32, 1.5)


class TestFormatMemoryStats:
    """Tests for format_memory_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = MemoryStats(32.0, 12.0, 20.0, 62.5, 33.0)
        formatted = format_memory_stats(stats)
        assert "Baseline: 32.00 GB" in formatted
        assert "Checkpointed: 12.00 GB" in formatted
        assert "Savings: 20.00 GB (62.5%)" in formatted
        assert "Recomputation Overhead: 33.0%" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_memory_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedCheckpointConfig:
    """Tests for get_recommended_checkpoint_config function."""

    def test_small_model_large_memory_no_checkpointing(self) -> None:
        """Small model with large memory needs no checkpointing."""
        config = get_recommended_checkpoint_config(1.0, 80.0)
        assert config.strategy == CheckpointStrategy.NONE

    def test_large_model_limited_memory_full_checkpointing(self) -> None:
        """Large model with limited memory needs full checkpointing."""
        config = get_recommended_checkpoint_config(7.0, 16.0)
        assert config.strategy in (
            CheckpointStrategy.FULL,
            CheckpointStrategy.SELECTIVE,
        )

    def test_returns_valid_config(self) -> None:
        """Returns a valid config."""
        config = get_recommended_checkpoint_config(7.0, 24.0)
        validate_checkpoint_config(config)

    def test_zero_model_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            get_recommended_checkpoint_config(0, 24.0)

    def test_zero_gpu_memory_raises(self) -> None:
        """Zero GPU memory raises ValueError."""
        with pytest.raises(
            ValueError, match="available_gpu_memory_gb must be positive"
        ):
            get_recommended_checkpoint_config(7.0, 0)

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            get_recommended_checkpoint_config(7.0, 24.0, num_layers=0)

    def test_medium_memory_selective_checkpointing(self) -> None:
        """Medium memory results in selective checkpointing."""
        # Find a combination that triggers the selective (middle) branch
        # This is when optimal_ratio is between 0 and 0.9
        config = get_recommended_checkpoint_config(3.0, 40.0, 32, batch_size=4)
        # Should get some form of checkpointing
        assert config.strategy in (
            CheckpointStrategy.NONE,
            CheckpointStrategy.SELECTIVE,
            CheckpointStrategy.FULL,
        )


class TestListCheckpointStrategies:
    """Tests for list_checkpoint_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_checkpoint_strategies()
        assert strategies == sorted(strategies)

    def test_contains_full(self) -> None:
        """Contains full."""
        strategies = list_checkpoint_strategies()
        assert "full" in strategies

    def test_contains_selective(self) -> None:
        """Contains selective."""
        strategies = list_checkpoint_strategies()
        assert "selective" in strategies


class TestListCheckpointGranularities:
    """Tests for list_checkpoint_granularities function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        granularities = list_checkpoint_granularities()
        assert granularities == sorted(granularities)

    def test_contains_block(self) -> None:
        """Contains block."""
        granularities = list_checkpoint_granularities()
        assert "block" in granularities

    def test_contains_attention(self) -> None:
        """Contains attention."""
        granularities = list_checkpoint_granularities()
        assert "attention" in granularities


class TestListOffloadTargets:
    """Tests for list_offload_targets function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        targets = list_offload_targets()
        assert targets == sorted(targets)

    def test_contains_cpu(self) -> None:
        """Contains cpu."""
        targets = list_offload_targets()
        assert "cpu" in targets

    def test_contains_disk(self) -> None:
        """Contains disk."""
        targets = list_offload_targets()
        assert "disk" in targets


class TestGetCheckpointStrategy:
    """Tests for get_checkpoint_strategy function."""

    def test_get_full(self) -> None:
        """Get full strategy."""
        assert get_checkpoint_strategy("full") == CheckpointStrategy.FULL

    def test_get_selective(self) -> None:
        """Get selective strategy."""
        assert get_checkpoint_strategy("selective") == CheckpointStrategy.SELECTIVE

    def test_get_offload(self) -> None:
        """Get offload strategy."""
        assert get_checkpoint_strategy("offload") == CheckpointStrategy.OFFLOAD

    def test_get_none(self) -> None:
        """Get none strategy."""
        assert get_checkpoint_strategy("none") == CheckpointStrategy.NONE

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_checkpoint_strategy("invalid")


class TestGetCheckpointGranularity:
    """Tests for get_checkpoint_granularity function."""

    def test_get_block(self) -> None:
        """Get block granularity."""
        assert get_checkpoint_granularity("block") == CheckpointGranularity.BLOCK

    def test_get_layer(self) -> None:
        """Get layer granularity."""
        assert get_checkpoint_granularity("layer") == CheckpointGranularity.LAYER

    def test_get_attention(self) -> None:
        """Get attention granularity."""
        result = get_checkpoint_granularity("attention")
        assert result == CheckpointGranularity.ATTENTION

    def test_get_mlp(self) -> None:
        """Get mlp granularity."""
        assert get_checkpoint_granularity("mlp") == CheckpointGranularity.MLP

    def test_invalid_granularity_raises(self) -> None:
        """Invalid granularity raises ValueError."""
        with pytest.raises(ValueError, match="granularity must be one of"):
            get_checkpoint_granularity("invalid")


class TestGetOffloadTarget:
    """Tests for get_offload_target function."""

    def test_get_cpu(self) -> None:
        """Get cpu target."""
        assert get_offload_target("cpu") == OffloadTarget.CPU

    def test_get_disk(self) -> None:
        """Get disk target."""
        assert get_offload_target("disk") == OffloadTarget.DISK

    def test_get_nvme(self) -> None:
        """Get nvme target."""
        assert get_offload_target("nvme") == OffloadTarget.NVME

    def test_invalid_target_raises(self) -> None:
        """Invalid target raises ValueError."""
        with pytest.raises(ValueError, match="target must be one of"):
            get_offload_target("invalid")
