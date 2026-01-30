"""Tests for training.parallelism module."""

from __future__ import annotations

import pytest

from hf_gtc.training.parallelism import (
    VALID_COMMUNICATION_BACKENDS,
    VALID_PARALLEL_SHARDING_STRATEGIES,
    VALID_PARALLELISM_TYPES,
    CommunicationBackend,
    ParallelConfig,
    ParallelFSDPConfig,
    ParallelismType,
    ParallelShardingStrategy,
    ParallelStats,
    PipelineParallelConfig,
    TensorParallelConfig,
    calculate_memory_per_device,
    calculate_world_size,
    create_fsdp_config,
    create_parallel_config,
    create_parallel_stats,
    create_pipeline_parallel_config,
    create_tensor_parallel_config,
    estimate_communication_overhead,
    format_parallel_stats,
    get_communication_backend,
    get_parallelism_type,
    get_recommended_parallel_config,
    get_sharding_strategy,
    list_communication_backends,
    list_parallelism_types,
    list_sharding_strategies,
    optimize_parallelism_strategy,
    validate_fsdp_config,
    validate_parallel_config,
    validate_pipeline_parallel_config,
    validate_tensor_parallel_config,
)


class TestParallelismType:
    """Tests for ParallelismType enum."""

    def test_all_types_have_values(self) -> None:
        """All parallelism types have string values."""
        for pt in ParallelismType:
            assert isinstance(pt.value, str)

    def test_data_value(self) -> None:
        """DATA has correct value."""
        assert ParallelismType.DATA.value == "data"

    def test_tensor_value(self) -> None:
        """TENSOR has correct value."""
        assert ParallelismType.TENSOR.value == "tensor"

    def test_pipeline_value(self) -> None:
        """PIPELINE has correct value."""
        assert ParallelismType.PIPELINE.value == "pipeline"

    def test_sequence_value(self) -> None:
        """SEQUENCE has correct value."""
        assert ParallelismType.SEQUENCE.value == "sequence"

    def test_expert_value(self) -> None:
        """EXPERT has correct value."""
        assert ParallelismType.EXPERT.value == "expert"

    def test_valid_parallelism_types_frozenset(self) -> None:
        """VALID_PARALLELISM_TYPES is a frozenset."""
        assert isinstance(VALID_PARALLELISM_TYPES, frozenset)
        assert len(VALID_PARALLELISM_TYPES) == 5


class TestParallelShardingStrategy:
    """Tests for ParallelShardingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All sharding strategies have string values."""
        for ss in ParallelShardingStrategy:
            assert isinstance(ss.value, str)

    def test_full_shard_value(self) -> None:
        """FULL_SHARD has correct value."""
        assert ParallelShardingStrategy.FULL_SHARD.value == "full_shard"

    def test_shard_grad_op_value(self) -> None:
        """SHARD_GRAD_OP has correct value."""
        assert ParallelShardingStrategy.SHARD_GRAD_OP.value == "shard_grad_op"

    def test_no_shard_value(self) -> None:
        """NO_SHARD has correct value."""
        assert ParallelShardingStrategy.NO_SHARD.value == "no_shard"

    def test_hybrid_value(self) -> None:
        """HYBRID has correct value."""
        assert ParallelShardingStrategy.HYBRID.value == "hybrid"

    def test_valid_sharding_strategies_frozenset(self) -> None:
        """VALID_PARALLEL_SHARDING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_PARALLEL_SHARDING_STRATEGIES, frozenset)
        assert len(VALID_PARALLEL_SHARDING_STRATEGIES) == 4


class TestCommunicationBackend:
    """Tests for CommunicationBackend enum."""

    def test_all_backends_have_values(self) -> None:
        """All backends have string values."""
        for cb in CommunicationBackend:
            assert isinstance(cb.value, str)

    def test_nccl_value(self) -> None:
        """NCCL has correct value."""
        assert CommunicationBackend.NCCL.value == "nccl"

    def test_gloo_value(self) -> None:
        """GLOO has correct value."""
        assert CommunicationBackend.GLOO.value == "gloo"

    def test_mpi_value(self) -> None:
        """MPI has correct value."""
        assert CommunicationBackend.MPI.value == "mpi"

    def test_valid_communication_backends_frozenset(self) -> None:
        """VALID_COMMUNICATION_BACKENDS is a frozenset."""
        assert isinstance(VALID_COMMUNICATION_BACKENDS, frozenset)
        assert len(VALID_COMMUNICATION_BACKENDS) == 3


class TestTensorParallelConfig:
    """Tests for TensorParallelConfig dataclass."""

    def test_create_config(self) -> None:
        """Create tensor parallel config."""
        config = TensorParallelConfig(
            tp_size=8,
            partition_dim=1,
            sequence_parallel=True,
            async_comm=True,
        )
        assert config.tp_size == 8
        assert config.partition_dim == 1
        assert config.sequence_parallel is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = TensorParallelConfig(8, 1, True, True)
        with pytest.raises(AttributeError):
            config.tp_size = 16  # type: ignore[misc]


class TestPipelineParallelConfig:
    """Tests for PipelineParallelConfig dataclass."""

    def test_create_config(self) -> None:
        """Create pipeline parallel config."""
        config = PipelineParallelConfig(
            pp_size=4,
            num_microbatches=8,
            interleave=True,
            activation_checkpointing=True,
        )
        assert config.pp_size == 4
        assert config.num_microbatches == 8
        assert config.interleave is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PipelineParallelConfig(4, 8, True, True)
        with pytest.raises(AttributeError):
            config.pp_size = 8  # type: ignore[misc]


class TestParallelFSDPConfig:
    """Tests for ParallelFSDPConfig dataclass."""

    def test_create_config(self) -> None:
        """Create FSDP config."""
        config = ParallelFSDPConfig(
            sharding_strategy=ParallelShardingStrategy.FULL_SHARD,
            cpu_offload=False,
            backward_prefetch=True,
            mixed_precision=True,
        )
        assert config.sharding_strategy == ParallelShardingStrategy.FULL_SHARD
        assert config.cpu_offload is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, False, True, True
        )
        with pytest.raises(AttributeError):
            config.cpu_offload = True  # type: ignore[misc]


class TestParallelConfig:
    """Tests for ParallelConfig dataclass."""

    def test_create_config(self) -> None:
        """Create parallel config."""
        tp = TensorParallelConfig(2, 1, True, True)
        pp = PipelineParallelConfig(2, 4, False, True)
        fsdp = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, False, True, True
        )
        config = ParallelConfig(
            dp_size=2,
            tp_config=tp,
            pp_config=pp,
            fsdp_config=fsdp,
            backend=CommunicationBackend.NCCL,
        )
        assert config.dp_size == 2
        assert config.backend == CommunicationBackend.NCCL

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        tp = TensorParallelConfig(2, 1, True, True)
        pp = PipelineParallelConfig(2, 4, False, True)
        fsdp = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, False, True, True
        )
        config = ParallelConfig(2, tp, pp, fsdp, CommunicationBackend.NCCL)
        with pytest.raises(AttributeError):
            config.dp_size = 4  # type: ignore[misc]


class TestParallelStats:
    """Tests for ParallelStats dataclass."""

    def test_create_stats(self) -> None:
        """Create parallel stats."""
        stats = ParallelStats(
            world_size=16,
            memory_per_device_gb=40.0,
            communication_overhead=0.15,
            efficiency=0.85,
        )
        assert stats.world_size == 16
        assert stats.efficiency == 0.85

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = ParallelStats(16, 40.0, 0.15, 0.85)
        with pytest.raises(AttributeError):
            stats.world_size = 32  # type: ignore[misc]


class TestValidateTensorParallelConfig:
    """Tests for validate_tensor_parallel_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = TensorParallelConfig(8, 1, True, True)
        validate_tensor_parallel_config(config)

    def test_tp_size_one_valid(self) -> None:
        """tp_size of 1 is valid (no tensor parallelism)."""
        config = TensorParallelConfig(1, 1, False, True)
        validate_tensor_parallel_config(config)

    def test_zero_tp_size_raises(self) -> None:
        """Zero tp_size raises ValueError."""
        config = TensorParallelConfig(0, 1, True, True)
        with pytest.raises(ValueError, match="tp_size must be positive"):
            validate_tensor_parallel_config(config)

    def test_negative_tp_size_raises(self) -> None:
        """Negative tp_size raises ValueError."""
        config = TensorParallelConfig(-1, 1, True, True)
        with pytest.raises(ValueError, match="tp_size must be positive"):
            validate_tensor_parallel_config(config)

    def test_invalid_partition_dim_raises(self) -> None:
        """Invalid partition_dim raises ValueError."""
        config = TensorParallelConfig(8, 2, True, True)
        with pytest.raises(ValueError, match="partition_dim must be 0 or 1"):
            validate_tensor_parallel_config(config)

    def test_non_power_of_2_tp_size_raises(self) -> None:
        """Non-power-of-2 tp_size raises ValueError."""
        config = TensorParallelConfig(6, 1, True, True)
        with pytest.raises(ValueError, match="tp_size must be a power of 2"):
            validate_tensor_parallel_config(config)

    def test_power_of_2_tp_sizes(self) -> None:
        """Power of 2 tp_sizes are valid."""
        for tp_size in [1, 2, 4, 8, 16, 32]:
            config = TensorParallelConfig(tp_size, 1, True, True)
            validate_tensor_parallel_config(config)


class TestValidatePipelineParallelConfig:
    """Tests for validate_pipeline_parallel_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PipelineParallelConfig(4, 8, True, True)
        validate_pipeline_parallel_config(config)

    def test_zero_pp_size_raises(self) -> None:
        """Zero pp_size raises ValueError."""
        config = PipelineParallelConfig(0, 8, True, True)
        with pytest.raises(ValueError, match="pp_size must be positive"):
            validate_pipeline_parallel_config(config)

    def test_negative_pp_size_raises(self) -> None:
        """Negative pp_size raises ValueError."""
        config = PipelineParallelConfig(-1, 8, True, True)
        with pytest.raises(ValueError, match="pp_size must be positive"):
            validate_pipeline_parallel_config(config)

    def test_zero_microbatches_raises(self) -> None:
        """Zero num_microbatches raises ValueError."""
        config = PipelineParallelConfig(4, 0, True, True)
        with pytest.raises(ValueError, match="num_microbatches must be positive"):
            validate_pipeline_parallel_config(config)

    def test_microbatches_less_than_pp_size_raises(self) -> None:
        """num_microbatches < pp_size raises ValueError."""
        config = PipelineParallelConfig(8, 4, True, True)
        with pytest.raises(ValueError, match=r"num_microbatches.*must be >= pp_size"):
            validate_pipeline_parallel_config(config)


class TestValidateParallelFSDPConfig:
    """Tests for validate_fsdp_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, False, True, True
        )
        validate_fsdp_config(config)

    def test_cpu_offload_with_no_shard_raises(self) -> None:
        """CPU offload with NO_SHARD raises ValueError."""
        config = ParallelFSDPConfig(ParallelShardingStrategy.NO_SHARD, True, True, True)
        with pytest.raises(ValueError, match="cpu_offload requires sharding"):
            validate_fsdp_config(config)

    def test_cpu_offload_with_full_shard_valid(self) -> None:
        """CPU offload with FULL_SHARD is valid."""
        config = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, True, True, True
        )
        validate_fsdp_config(config)


class TestValidateParallelConfig:
    """Tests for validate_parallel_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        tp = TensorParallelConfig(2, 1, True, True)
        pp = PipelineParallelConfig(2, 4, False, True)
        fsdp = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, False, True, True
        )
        config = ParallelConfig(2, tp, pp, fsdp, CommunicationBackend.NCCL)
        validate_parallel_config(config)

    def test_zero_dp_size_raises(self) -> None:
        """Zero dp_size raises ValueError."""
        tp = TensorParallelConfig(2, 1, True, True)
        pp = PipelineParallelConfig(2, 4, False, True)
        fsdp = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, False, True, True
        )
        config = ParallelConfig(0, tp, pp, fsdp, CommunicationBackend.NCCL)
        with pytest.raises(ValueError, match="dp_size must be positive"):
            validate_parallel_config(config)

    def test_invalid_tp_config_raises(self) -> None:
        """Invalid tp_config raises ValueError."""
        tp = TensorParallelConfig(0, 1, True, True)  # Invalid
        pp = PipelineParallelConfig(2, 4, False, True)
        fsdp = ParallelFSDPConfig(
            ParallelShardingStrategy.FULL_SHARD, False, True, True
        )
        config = ParallelConfig(2, tp, pp, fsdp, CommunicationBackend.NCCL)
        with pytest.raises(ValueError, match="tp_size must be positive"):
            validate_parallel_config(config)


class TestCreateTensorParallelConfig:
    """Tests for create_tensor_parallel_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_tensor_parallel_config()
        assert config.tp_size == 1
        assert config.partition_dim == 1
        assert config.sequence_parallel is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_tensor_parallel_config(
            tp_size=8,
            sequence_parallel=True,
        )
        assert config.tp_size == 8
        assert config.sequence_parallel is True

    def test_partition_dim_zero(self) -> None:
        """Create with partition_dim=0."""
        config = create_tensor_parallel_config(partition_dim=0)
        assert config.partition_dim == 0

    def test_invalid_tp_size_raises(self) -> None:
        """Invalid tp_size raises ValueError."""
        with pytest.raises(ValueError, match="tp_size must be positive"):
            create_tensor_parallel_config(tp_size=0)


class TestCreatePipelineParallelConfig:
    """Tests for create_pipeline_parallel_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_pipeline_parallel_config()
        assert config.pp_size == 1
        assert config.num_microbatches == 1
        assert config.activation_checkpointing is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_pipeline_parallel_config(
            pp_size=4,
            num_microbatches=8,
            interleave=True,
        )
        assert config.pp_size == 4
        assert config.num_microbatches == 8
        assert config.interleave is True

    def test_invalid_pp_size_raises(self) -> None:
        """Invalid pp_size raises ValueError."""
        with pytest.raises(ValueError, match="pp_size must be positive"):
            create_pipeline_parallel_config(pp_size=0)


class TestCreateParallelFSDPConfig:
    """Tests for create_fsdp_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_fsdp_config()
        assert config.sharding_strategy == ParallelShardingStrategy.FULL_SHARD
        assert config.backward_prefetch is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_fsdp_config(
            sharding_strategy="shard_grad_op",
            cpu_offload=False,
        )
        assert config.sharding_strategy == ParallelShardingStrategy.SHARD_GRAD_OP

    def test_with_enum_strategy(self) -> None:
        """Create with enum sharding strategy."""
        config = create_fsdp_config(sharding_strategy=ParallelShardingStrategy.HYBRID)
        assert config.sharding_strategy == ParallelShardingStrategy.HYBRID

    def test_invalid_strategy_raises(self) -> None:
        """Invalid sharding strategy raises ValueError."""
        with pytest.raises(ValueError, match="sharding_strategy must be one of"):
            create_fsdp_config(sharding_strategy="invalid")

    @pytest.mark.parametrize(
        "strategy",
        ["full_shard", "shard_grad_op", "no_shard", "hybrid"],
    )
    def test_all_strategies(self, strategy: str) -> None:
        """Test all sharding strategies."""
        config = create_fsdp_config(sharding_strategy=strategy)
        assert config.sharding_strategy.value == strategy


class TestCreateParallelConfig:
    """Tests for create_parallel_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_parallel_config()
        assert config.dp_size == 1
        assert config.tp_config.tp_size == 1
        assert config.pp_config.pp_size == 1
        assert config.backend == CommunicationBackend.NCCL

    def test_custom_config(self) -> None:
        """Create custom config."""
        tp = create_tensor_parallel_config(tp_size=8)
        config = create_parallel_config(dp_size=2, tp_config=tp)
        assert config.dp_size == 2
        assert config.tp_config.tp_size == 8

    def test_with_string_backend(self) -> None:
        """Create with string backend."""
        config = create_parallel_config(backend="gloo")
        assert config.backend == CommunicationBackend.GLOO

    def test_with_all_custom_configs(self) -> None:
        """Create with all custom configs."""
        tp = create_tensor_parallel_config(tp_size=4)
        pp = create_pipeline_parallel_config(pp_size=2, num_microbatches=4)
        fsdp = create_fsdp_config(sharding_strategy="shard_grad_op")
        config = create_parallel_config(
            dp_size=4,
            tp_config=tp,
            pp_config=pp,
            fsdp_config=fsdp,
            backend="mpi",
        )
        assert config.dp_size == 4
        assert config.tp_config.tp_size == 4
        assert config.pp_config.pp_size == 2
        assert config.backend == CommunicationBackend.MPI

    def test_invalid_dp_size_raises(self) -> None:
        """Invalid dp_size raises ValueError."""
        with pytest.raises(ValueError, match="dp_size must be positive"):
            create_parallel_config(dp_size=0)

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="communication_backend must be one of"):
            create_parallel_config(backend="invalid")


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_parallelism_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_parallelism_types()
        assert types == sorted(types)
        assert "tensor" in types

    def test_list_parallelism_types_complete(self) -> None:
        """Returns all parallelism types."""
        types = list_parallelism_types()
        assert len(types) == 5

    def test_list_sharding_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_sharding_strategies()
        assert strategies == sorted(strategies)
        assert "full_shard" in strategies

    def test_list_sharding_strategies_complete(self) -> None:
        """Returns all sharding strategies."""
        strategies = list_sharding_strategies()
        assert len(strategies) == 4

    def test_list_communication_backends_sorted(self) -> None:
        """Returns sorted list."""
        backends = list_communication_backends()
        assert backends == sorted(backends)
        assert "nccl" in backends

    def test_list_communication_backends_complete(self) -> None:
        """Returns all backends."""
        backends = list_communication_backends()
        assert len(backends) == 3


class TestGetParallelismType:
    """Tests for get_parallelism_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("data", ParallelismType.DATA),
            ("tensor", ParallelismType.TENSOR),
            ("pipeline", ParallelismType.PIPELINE),
            ("sequence", ParallelismType.SEQUENCE),
            ("expert", ParallelismType.EXPERT),
        ],
    )
    def test_all_types(self, name: str, expected: ParallelismType) -> None:
        """Test all valid parallelism types."""
        assert get_parallelism_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="parallelism_type must be one of"):
            get_parallelism_type("invalid")


class TestGetParallelShardingStrategy:
    """Tests for get_sharding_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("full_shard", ParallelShardingStrategy.FULL_SHARD),
            ("shard_grad_op", ParallelShardingStrategy.SHARD_GRAD_OP),
            ("no_shard", ParallelShardingStrategy.NO_SHARD),
            ("hybrid", ParallelShardingStrategy.HYBRID),
        ],
    )
    def test_all_strategies(
        self, name: str, expected: ParallelShardingStrategy
    ) -> None:
        """Test all valid sharding strategies."""
        assert get_sharding_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="sharding_strategy must be one of"):
            get_sharding_strategy("invalid")


class TestGetCommunicationBackend:
    """Tests for get_communication_backend function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("nccl", CommunicationBackend.NCCL),
            ("gloo", CommunicationBackend.GLOO),
            ("mpi", CommunicationBackend.MPI),
        ],
    )
    def test_all_backends(self, name: str, expected: CommunicationBackend) -> None:
        """Test all valid communication backends."""
        assert get_communication_backend(name) == expected

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="communication_backend must be one of"):
            get_communication_backend("invalid")


class TestCalculateWorldSize:
    """Tests for calculate_world_size function."""

    def test_default_config(self) -> None:
        """Default config has world_size 1."""
        config = create_parallel_config()
        assert calculate_world_size(config) == 1

    def test_data_parallel_only(self) -> None:
        """Data parallel only."""
        config = create_parallel_config(dp_size=8)
        assert calculate_world_size(config) == 8

    def test_tensor_parallel_only(self) -> None:
        """Tensor parallel only."""
        tp = create_tensor_parallel_config(tp_size=4)
        config = create_parallel_config(tp_config=tp)
        assert calculate_world_size(config) == 4

    def test_pipeline_parallel_only(self) -> None:
        """Pipeline parallel only."""
        pp = create_pipeline_parallel_config(pp_size=4, num_microbatches=4)
        config = create_parallel_config(pp_config=pp)
        assert calculate_world_size(config) == 4

    def test_combined_parallelism(self) -> None:
        """Combined DP + TP + PP."""
        tp = create_tensor_parallel_config(tp_size=4)
        pp = create_pipeline_parallel_config(pp_size=2, num_microbatches=4)
        config = create_parallel_config(dp_size=2, tp_config=tp, pp_config=pp)
        assert calculate_world_size(config) == 2 * 4 * 2  # 16


class TestEstimateCommunicationOverhead:
    """Tests for estimate_communication_overhead function."""

    def test_default_config(self) -> None:
        """Default config has minimal overhead."""
        config = create_parallel_config()
        overhead = estimate_communication_overhead(config)
        assert 0.0 <= overhead <= 0.5

    def test_data_parallel_overhead(self) -> None:
        """Data parallel has some overhead."""
        config = create_parallel_config(dp_size=8)
        overhead = estimate_communication_overhead(config)
        assert overhead > 0.0

    def test_tensor_parallel_high_overhead(self) -> None:
        """Tensor parallel has higher overhead."""
        tp = create_tensor_parallel_config(tp_size=8)
        config = create_parallel_config(tp_config=tp)
        overhead = estimate_communication_overhead(config)
        assert overhead > 0.1

    def test_sequence_parallel_reduces_overhead(self) -> None:
        """Sequence parallel reduces TP overhead."""
        tp_no_sp = create_tensor_parallel_config(tp_size=8, sequence_parallel=False)
        tp_sp = create_tensor_parallel_config(tp_size=8, sequence_parallel=True)
        config_no_sp = create_parallel_config(tp_config=tp_no_sp)
        config_sp = create_parallel_config(tp_config=tp_sp)
        assert estimate_communication_overhead(
            config_sp
        ) < estimate_communication_overhead(config_no_sp)

    def test_larger_models_less_overhead(self) -> None:
        """Larger models amortize overhead better."""
        config = create_parallel_config(dp_size=8)
        small_overhead = estimate_communication_overhead(
            config, model_params_billions=1.0
        )
        large_overhead = estimate_communication_overhead(
            config, model_params_billions=70.0
        )
        assert large_overhead < small_overhead

    def test_zero_model_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        config = create_parallel_config()
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            estimate_communication_overhead(config, model_params_billions=0)

    def test_interleave_reduces_pp_overhead(self) -> None:
        """Interleaved pipeline reduces overhead."""
        pp_no_interleave = create_pipeline_parallel_config(
            pp_size=4, num_microbatches=8, interleave=False
        )
        pp_interleave = create_pipeline_parallel_config(
            pp_size=4, num_microbatches=8, interleave=True
        )
        config_no = create_parallel_config(pp_config=pp_no_interleave)
        config_yes = create_parallel_config(pp_config=pp_interleave)
        assert estimate_communication_overhead(
            config_yes
        ) < estimate_communication_overhead(config_no)


class TestCalculateMemoryPerDevice:
    """Tests for calculate_memory_per_device function."""

    def test_default_config(self) -> None:
        """Default config memory calculation."""
        config = create_parallel_config()
        memory = calculate_memory_per_device(config)
        assert memory > 0

    def test_tensor_parallel_reduces_memory(self) -> None:
        """Tensor parallel reduces memory per device."""
        config_1 = create_parallel_config()
        tp = create_tensor_parallel_config(tp_size=8)
        config_8 = create_parallel_config(tp_config=tp)
        memory_1 = calculate_memory_per_device(config_1)
        memory_8 = calculate_memory_per_device(config_8)
        assert memory_8 < memory_1

    def test_fsdp_reduces_memory(self) -> None:
        """FSDP sharding reduces memory."""
        fsdp_no_shard = create_fsdp_config(sharding_strategy="no_shard")
        fsdp_full = create_fsdp_config(sharding_strategy="full_shard")
        config_no = create_parallel_config(dp_size=8, fsdp_config=fsdp_no_shard)
        config_full = create_parallel_config(dp_size=8, fsdp_config=fsdp_full)
        memory_no = calculate_memory_per_device(config_no)
        memory_full = calculate_memory_per_device(config_full)
        assert memory_full < memory_no

    def test_activation_checkpointing_reduces_memory(self) -> None:
        """Activation checkpointing reduces memory."""
        pp_no_ckpt = create_pipeline_parallel_config(activation_checkpointing=False)
        pp_ckpt = create_pipeline_parallel_config(activation_checkpointing=True)
        config_no = create_parallel_config(pp_config=pp_no_ckpt)
        config_yes = create_parallel_config(pp_config=pp_ckpt)
        memory_no = calculate_memory_per_device(config_no)
        memory_yes = calculate_memory_per_device(config_yes)
        assert memory_yes < memory_no

    def test_zero_model_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        config = create_parallel_config()
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            calculate_memory_per_device(config, model_params_billions=0)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        config = create_parallel_config()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_memory_per_device(config, batch_size=0)

    def test_zero_sequence_length_raises(self) -> None:
        """Zero sequence length raises ValueError."""
        config = create_parallel_config()
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            calculate_memory_per_device(config, sequence_length=0)


class TestOptimizeParallelismStrategy:
    """Tests for optimize_parallelism_strategy function."""

    def test_small_model_uses_dp(self) -> None:
        """Small model uses data parallelism."""
        config = optimize_parallelism_strategy(1.0, 8)
        assert config.dp_size >= 1
        assert config.tp_config.tp_size <= 2

    def test_large_model_uses_tp(self) -> None:
        """Large model uses tensor parallelism."""
        config = optimize_parallelism_strategy(70.0, 64)
        assert config.tp_config.tp_size > 1

    def test_uses_all_gpus(self) -> None:
        """Uses available GPUs efficiently."""
        config = optimize_parallelism_strategy(7.0, 8)
        world_size = calculate_world_size(config)
        assert world_size <= 8

    def test_zero_model_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            optimize_parallelism_strategy(0, 8)

    def test_zero_num_gpus_raises(self) -> None:
        """Zero num_gpus raises ValueError."""
        with pytest.raises(ValueError, match="num_gpus must be positive"):
            optimize_parallelism_strategy(7.0, 0)

    def test_zero_gpu_memory_raises(self) -> None:
        """Zero gpu_memory raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_gb must be positive"):
            optimize_parallelism_strategy(7.0, 8, gpu_memory_gb=0)

    def test_zero_target_batch_raises(self) -> None:
        """Zero target_batch_size raises ValueError."""
        with pytest.raises(ValueError, match="target_batch_size must be positive"):
            optimize_parallelism_strategy(7.0, 8, target_batch_size=0)

    def test_valid_config_returned(self) -> None:
        """Returned config is valid."""
        config = optimize_parallelism_strategy(7.0, 8)
        validate_parallel_config(config)

    def test_sequence_parallel_with_tp(self) -> None:
        """Sequence parallel enabled when using tensor parallel."""
        config = optimize_parallelism_strategy(70.0, 64)
        if config.tp_config.tp_size > 1:
            assert config.tp_config.sequence_parallel is True


class TestFormatParallelStats:
    """Tests for format_parallel_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = ParallelStats(16, 40.0, 0.15, 0.85)
        formatted = format_parallel_stats(stats)
        assert "World Size: 16" in formatted
        assert "Memory: 40.00 GB" in formatted
        assert "Communication Overhead: 15.0%" in formatted
        assert "Efficiency: 85.0%" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = ParallelStats(8, 20.0, 0.1, 0.9)
        formatted = format_parallel_stats(stats)
        assert "World Size:" in formatted
        assert "Memory:" in formatted
        assert "Communication Overhead:" in formatted
        assert "Efficiency:" in formatted

    def test_contains_header(self) -> None:
        """Formatted string has header."""
        stats = ParallelStats(1, 10.0, 0.0, 1.0)
        formatted = format_parallel_stats(stats)
        assert "Parallel Stats:" in formatted


class TestGetRecommendedParallelConfig:
    """Tests for get_recommended_parallel_config function."""

    def test_small_model(self) -> None:
        """Config for small model."""
        config = get_recommended_parallel_config(1.0, 4)
        validate_parallel_config(config)
        assert calculate_world_size(config) <= 4

    def test_medium_model(self) -> None:
        """Config for medium model."""
        config = get_recommended_parallel_config(7.0, 8)
        validate_parallel_config(config)
        assert calculate_world_size(config) <= 8

    def test_large_model(self) -> None:
        """Config for large model."""
        config = get_recommended_parallel_config(70.0, 64)
        validate_parallel_config(config)

    def test_zero_model_params_raises(self) -> None:
        """Zero model_params raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            get_recommended_parallel_config(0, 8)

    def test_zero_num_gpus_raises(self) -> None:
        """Zero num_gpus raises ValueError."""
        with pytest.raises(ValueError, match="num_gpus must be positive"):
            get_recommended_parallel_config(7.0, 0)

    def test_zero_gpu_memory_raises(self) -> None:
        """Zero gpu_memory raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_gb must be positive"):
            get_recommended_parallel_config(7.0, 8, gpu_memory_gb=0)


class TestCreateParallelStats:
    """Tests for create_parallel_stats function."""

    def test_default_config(self) -> None:
        """Stats for default config."""
        config = create_parallel_config()
        stats = create_parallel_stats(config)
        assert stats.world_size == 1
        assert stats.memory_per_device_gb > 0
        assert stats.efficiency <= 1.0

    def test_combined_parallelism(self) -> None:
        """Stats for combined parallelism."""
        tp = create_tensor_parallel_config(tp_size=4)
        pp = create_pipeline_parallel_config(pp_size=2, num_microbatches=4)
        config = create_parallel_config(dp_size=2, tp_config=tp, pp_config=pp)
        stats = create_parallel_stats(config)
        assert stats.world_size == 16
        assert 0.0 <= stats.communication_overhead <= 0.5

    def test_zero_model_params_raises(self) -> None:
        """Zero model_params raises ValueError."""
        config = create_parallel_config()
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            create_parallel_stats(config, model_params_billions=0)

    def test_efficiency_matches_overhead(self) -> None:
        """Efficiency = 1 - overhead."""
        config = create_parallel_config(dp_size=4)
        stats = create_parallel_stats(config)
        assert stats.efficiency == pytest.approx(1.0 - stats.communication_overhead)
