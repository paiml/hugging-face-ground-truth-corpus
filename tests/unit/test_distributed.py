"""Tests for training.distributed module."""

from __future__ import annotations

import pytest

from hf_gtc.training.distributed import (
    VALID_BACKENDS,
    VALID_CHECKPOINTING,
    VALID_DEEPSPEED_STAGES,
    VALID_SHARDING_STRATEGIES,
    ActivationCheckpointing,
    DeepSpeedConfig,
    DeepSpeedStage,
    DistributedBackend,
    DistributedConfig,
    FSDPConfig,
    MemoryEstimate,
    ScalingMetrics,
    ShardingStrategy,
    calculate_scaling_efficiency,
    create_deepspeed_config,
    create_distributed_config,
    create_fsdp_config,
    estimate_deepspeed_memory,
    estimate_fsdp_memory,
    format_memory_estimate,
    get_deepspeed_stage,
    get_recommended_strategy,
    get_sharding_strategy,
    list_backends,
    list_deepspeed_stages,
    list_sharding_strategies,
    validate_deepspeed_config,
    validate_distributed_config,
    validate_fsdp_config,
)


class TestShardingStrategy:
    """Tests for ShardingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ShardingStrategy:
            assert isinstance(strategy.value, str)

    def test_full_shard_value(self) -> None:
        """Full shard has correct value."""
        assert ShardingStrategy.FULL_SHARD.value == "full_shard"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_SHARDING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_SHARDING_STRATEGIES, frozenset)


class TestDeepSpeedStage:
    """Tests for DeepSpeedStage enum."""

    def test_all_stages_have_values(self) -> None:
        """All stages have string values."""
        for stage in DeepSpeedStage:
            assert isinstance(stage.value, str)

    def test_stage_2_value(self) -> None:
        """Stage 2 has correct value."""
        assert DeepSpeedStage.STAGE_2.value == "stage_2"

    def test_valid_stages_frozenset(self) -> None:
        """VALID_DEEPSPEED_STAGES is a frozenset."""
        assert isinstance(VALID_DEEPSPEED_STAGES, frozenset)


class TestDistributedBackend:
    """Tests for DistributedBackend enum."""

    def test_all_backends_have_values(self) -> None:
        """All backends have string values."""
        for backend in DistributedBackend:
            assert isinstance(backend.value, str)

    def test_nccl_value(self) -> None:
        """NCCL has correct value."""
        assert DistributedBackend.NCCL.value == "nccl"


class TestActivationCheckpointing:
    """Tests for ActivationCheckpointing enum."""

    def test_all_options_have_values(self) -> None:
        """All options have string values."""
        for option in ActivationCheckpointing:
            assert isinstance(option.value, str)

    def test_full_value(self) -> None:
        """Full has correct value."""
        assert ActivationCheckpointing.FULL.value == "full"


class TestFSDPConfig:
    """Tests for FSDPConfig dataclass."""

    def test_create_config(self) -> None:
        """Create FSDP config."""
        config = FSDPConfig(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=False,
            mixed_precision=True,
            backward_prefetch=True,
            forward_prefetch=False,
            activation_checkpointing=ActivationCheckpointing.FULL,
        )
        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD


class TestValidateFSDPConfig:
    """Tests for validate_fsdp_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FSDPConfig(
            ShardingStrategy.FULL_SHARD,
            False,
            True,
            True,
            False,
            ActivationCheckpointing.FULL,
        )
        validate_fsdp_config(config)

    def test_cpu_offload_with_no_shard_raises(self) -> None:
        """CPU offload with NO_SHARD raises ValueError."""
        config = FSDPConfig(
            ShardingStrategy.NO_SHARD,
            True,
            True,
            True,
            False,
            ActivationCheckpointing.FULL,
        )
        with pytest.raises(ValueError, match="cpu_offload requires sharding"):
            validate_fsdp_config(config)


class TestDeepSpeedConfig:
    """Tests for DeepSpeedConfig dataclass."""

    def test_create_config(self) -> None:
        """Create DeepSpeed config."""
        config = DeepSpeedConfig(
            stage=DeepSpeedStage.STAGE_2,
            offload_optimizer=False,
            offload_param=False,
            overlap_comm=True,
            contiguous_gradients=True,
            reduce_bucket_size=500000000,
        )
        assert config.stage == DeepSpeedStage.STAGE_2


class TestValidateDeepSpeedConfig:
    """Tests for validate_deepspeed_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = DeepSpeedConfig(
            DeepSpeedStage.STAGE_2, False, False, True, True, 500000000
        )
        validate_deepspeed_config(config)

    def test_param_offload_without_stage3_raises(self) -> None:
        """Param offload without stage 3 raises ValueError."""
        config = DeepSpeedConfig(
            DeepSpeedStage.STAGE_1, False, True, True, True, 500000000
        )
        with pytest.raises(ValueError, match="offload_param requires ZeRO stage 3"):
            validate_deepspeed_config(config)

    def test_zero_bucket_size_raises(self) -> None:
        """Zero bucket size raises ValueError."""
        config = DeepSpeedConfig(DeepSpeedStage.STAGE_2, False, False, True, True, 0)
        with pytest.raises(ValueError, match="reduce_bucket_size must be positive"):
            validate_deepspeed_config(config)


class TestDistributedConfig:
    """Tests for DistributedConfig dataclass."""

    def test_create_config(self) -> None:
        """Create distributed config."""
        config = DistributedConfig(
            backend=DistributedBackend.NCCL,
            world_size=8,
            local_rank=0,
            num_nodes=2,
            gpus_per_node=4,
        )
        assert config.world_size == 8


class TestValidateDistributedConfig:
    """Tests for validate_distributed_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = DistributedConfig(DistributedBackend.NCCL, 8, 0, 2, 4)
        validate_distributed_config(config)

    def test_world_size_mismatch_raises(self) -> None:
        """World size mismatch raises ValueError."""
        config = DistributedConfig(DistributedBackend.NCCL, 8, 0, 2, 3)
        with pytest.raises(ValueError, match="world_size must equal"):
            validate_distributed_config(config)

    def test_zero_world_size_raises(self) -> None:
        """Zero world size raises ValueError."""
        config = DistributedConfig(DistributedBackend.NCCL, 0, 0, 1, 1)
        with pytest.raises(ValueError, match="world_size must be positive"):
            validate_distributed_config(config)

    def test_zero_num_nodes_raises(self) -> None:
        """Zero num_nodes raises ValueError."""
        config = DistributedConfig(DistributedBackend.NCCL, 1, 0, 0, 1)
        with pytest.raises(ValueError, match="num_nodes must be positive"):
            validate_distributed_config(config)

    def test_zero_gpus_per_node_raises(self) -> None:
        """Zero gpus_per_node raises ValueError."""
        config = DistributedConfig(DistributedBackend.NCCL, 1, 0, 1, 0)
        with pytest.raises(ValueError, match="gpus_per_node must be positive"):
            validate_distributed_config(config)

    def test_invalid_local_rank_raises(self) -> None:
        """Invalid local_rank raises ValueError."""
        config = DistributedConfig(DistributedBackend.NCCL, 4, 4, 1, 4)
        with pytest.raises(ValueError, match="local_rank must be in"):
            validate_distributed_config(config)


class TestCreateFSDPConfig:
    """Tests for create_fsdp_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_fsdp_config()
        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert config.mixed_precision is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_fsdp_config(
            sharding_strategy="shard_grad_op",
            activation_checkpointing="full",
        )
        assert config.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
        assert config.activation_checkpointing == ActivationCheckpointing.FULL

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="sharding_strategy must be one of"):
            create_fsdp_config(sharding_strategy="invalid")

    def test_invalid_checkpointing_raises(self) -> None:
        """Invalid checkpointing raises ValueError."""
        with pytest.raises(ValueError, match="activation_checkpointing must be one of"):
            create_fsdp_config(activation_checkpointing="invalid")


class TestCreateDeepSpeedConfig:
    """Tests for create_deepspeed_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_deepspeed_config()
        assert config.stage == DeepSpeedStage.STAGE_2
        assert config.overlap_comm is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_deepspeed_config(stage="stage_3", offload_param=True)
        assert config.stage == DeepSpeedStage.STAGE_3
        assert config.offload_param is True

    def test_invalid_stage_raises(self) -> None:
        """Invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="stage must be one of"):
            create_deepspeed_config(stage="invalid")


class TestCreateDistributedConfig:
    """Tests for create_distributed_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_distributed_config()
        assert config.world_size == 1
        assert config.backend == DistributedBackend.NCCL

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_distributed_config(world_size=8, num_nodes=2, gpus_per_node=4)
        assert config.world_size == 8
        assert config.num_nodes == 2

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be one of"):
            create_distributed_config(backend="invalid")


class TestEstimateFSDPMemory:
    """Tests for estimate_fsdp_memory function."""

    def test_basic_estimate(self) -> None:
        """Basic memory estimate."""
        est = estimate_fsdp_memory(7.0, world_size=8)
        assert est.total_memory_gb > 0

    def test_full_shard_less_memory(self) -> None:
        """Full shard uses less memory than no shard."""
        est_full = estimate_fsdp_memory(7.0, "full_shard", 8)
        est_no = estimate_fsdp_memory(7.0, "no_shard", 8)
        assert est_full.total_memory_gb < est_no.total_memory_gb


class TestEstimateDeepSpeedMemory:
    """Tests for estimate_deepspeed_memory function."""

    def test_basic_estimate(self) -> None:
        """Basic memory estimate."""
        est = estimate_deepspeed_memory(7.0, "stage_2", world_size=8)
        assert est.total_memory_gb > 0

    def test_stage_3_less_memory(self) -> None:
        """Stage 3 uses less memory than stage 2."""
        est_s2 = estimate_deepspeed_memory(7.0, "stage_2", 8)
        est_s3 = estimate_deepspeed_memory(7.0, "stage_3", 8)
        assert est_s3.total_memory_gb < est_s2.total_memory_gb


class TestCalculateScalingEfficiency:
    """Tests for calculate_scaling_efficiency function."""

    def test_perfect_scaling(self) -> None:
        """Perfect linear scaling."""
        efficiency = calculate_scaling_efficiency(100.0, 800.0, 8)
        assert efficiency == pytest.approx(1.0)

    def test_sub_linear_scaling(self) -> None:
        """Sub-linear scaling."""
        efficiency = calculate_scaling_efficiency(100.0, 750.0, 8)
        assert efficiency == pytest.approx(0.9375)

    def test_zero_single_gpu_raises(self) -> None:
        """Zero single GPU throughput raises ValueError."""
        with pytest.raises(ValueError, match="single_gpu_throughput must be positive"):
            calculate_scaling_efficiency(0, 800.0, 8)

    def test_zero_multi_gpu_raises(self) -> None:
        """Zero multi GPU throughput raises ValueError."""
        with pytest.raises(ValueError, match="multi_gpu_throughput must be positive"):
            calculate_scaling_efficiency(100.0, 0, 8)

    def test_zero_gpus_raises(self) -> None:
        """Zero GPUs raises ValueError."""
        with pytest.raises(ValueError, match="num_gpus must be positive"):
            calculate_scaling_efficiency(100.0, 800.0, 0)


class TestGetRecommendedStrategy:
    """Tests for get_recommended_strategy function."""

    def test_small_model_uses_ddp(self) -> None:
        """Small model uses DDP."""
        strategy = get_recommended_strategy(1.0, 80.0, 4)
        assert strategy == "ddp"

    def test_medium_model_uses_fsdp(self) -> None:
        """Medium model uses FSDP."""
        strategy = get_recommended_strategy(7.0, 80.0, 8)
        assert strategy == "fsdp_full_shard"

    def test_large_model_uses_deepspeed(self) -> None:
        """Large model uses DeepSpeed."""
        strategy = get_recommended_strategy(70.0, 80.0, 8)
        assert strategy == "deepspeed_stage_3"


class TestListShardingStrategies:
    """Tests for list_sharding_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_sharding_strategies()
        assert strategies == sorted(strategies)

    def test_contains_full_shard(self) -> None:
        """Contains full_shard."""
        strategies = list_sharding_strategies()
        assert "full_shard" in strategies


class TestListDeepSpeedStages:
    """Tests for list_deepspeed_stages function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        stages = list_deepspeed_stages()
        assert stages == sorted(stages)

    def test_contains_stage_2(self) -> None:
        """Contains stage_2."""
        stages = list_deepspeed_stages()
        assert "stage_2" in stages


class TestListBackends:
    """Tests for list_backends function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        backends = list_backends()
        assert backends == sorted(backends)

    def test_contains_nccl(self) -> None:
        """Contains nccl."""
        backends = list_backends()
        assert "nccl" in backends


class TestGetShardingStrategy:
    """Tests for get_sharding_strategy function."""

    def test_get_full_shard(self) -> None:
        """Get full_shard strategy."""
        assert get_sharding_strategy("full_shard") == ShardingStrategy.FULL_SHARD

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_sharding_strategy("invalid")


class TestGetDeepSpeedStage:
    """Tests for get_deepspeed_stage function."""

    def test_get_stage_2(self) -> None:
        """Get stage_2."""
        assert get_deepspeed_stage("stage_2") == DeepSpeedStage.STAGE_2

    def test_invalid_stage_raises(self) -> None:
        """Invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="stage must be one of"):
            get_deepspeed_stage("invalid")


class TestFormatMemoryEstimate:
    """Tests for format_memory_estimate function."""

    def test_basic_format(self) -> None:
        """Format basic estimate."""
        est = MemoryEstimate(7.0, 14.0, 7.0, 4.0, 32.0)
        formatted = format_memory_estimate(est)
        assert "Model: 7.00 GB" in formatted
        assert "Total: 32.00 GB" in formatted


class TestMemoryEstimate:
    """Tests for MemoryEstimate dataclass."""

    def test_create_estimate(self) -> None:
        """Create memory estimate."""
        est = MemoryEstimate(
            model_memory_gb=14.0,
            optimizer_memory_gb=28.0,
            gradient_memory_gb=14.0,
            activation_memory_gb=8.0,
            total_memory_gb=16.0,
        )
        assert est.total_memory_gb == pytest.approx(16.0)


class TestScalingMetrics:
    """Tests for ScalingMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Create scaling metrics."""
        metrics = ScalingMetrics(
            throughput_samples_per_sec=1000.0,
            scaling_efficiency=0.85,
            communication_overhead=0.15,
            gpu_utilization=0.92,
        )
        assert metrics.scaling_efficiency == pytest.approx(0.85)


class TestValidConstants:
    """Tests for validation constants."""

    def test_valid_backends_frozenset(self) -> None:
        """VALID_BACKENDS is a frozenset."""
        assert isinstance(VALID_BACKENDS, frozenset)

    def test_valid_checkpointing_frozenset(self) -> None:
        """VALID_CHECKPOINTING is a frozenset."""
        assert isinstance(VALID_CHECKPOINTING, frozenset)
