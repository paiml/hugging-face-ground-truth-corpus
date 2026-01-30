"""Tests for model compression utilities."""

from __future__ import annotations

import math

import pytest

from hf_gtc.deployment.compression import (
    CompressionConfig,
    CompressionStats,
    DecompositionMethod,
    FusionType,
    ImportanceMetric,
    LayerFusionConfig,
    LowRankConfig,
    PruningConfig,
    PruningMethod,
    PruningSchedule,
    PruningScope,
    StructuredPruningConfig,
    StructuredPruningDim,
    WeightSharingConfig,
    calculate_flops_reduction,
    calculate_low_rank_params,
    calculate_sparsity_at_step,
    calculate_weight_sharing_bits,
    create_compression_config,
    create_compression_stats,
    create_layer_fusion_config,
    create_low_rank_config,
    create_pruning_config,
    create_pruning_schedule,
    create_structured_pruning_config,
    create_weight_sharing_config,
    estimate_compressed_size,
    estimate_speedup_from_sparsity,
    format_compression_stats,
    get_decomposition_method,
    get_fusion_type,
    get_importance_metric,
    get_pruning_method,
    get_pruning_scope,
    get_recommended_compression_config,
    get_structured_pruning_dim,
    list_decomposition_methods,
    list_fusion_types,
    list_importance_metrics,
    list_pruning_methods,
    list_pruning_scopes,
    list_structured_pruning_dims,
    validate_compression_config,
    validate_low_rank_config,
    validate_pruning_config,
    validate_structured_pruning_config,
    validate_weight_sharing_config,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestPruningMethod:
    """Tests for PruningMethod enum."""

    def test_magnitude_value(self) -> None:
        """Test MAGNITUDE value."""
        assert PruningMethod.MAGNITUDE.value == "magnitude"

    def test_structured_value(self) -> None:
        """Test STRUCTURED value."""
        assert PruningMethod.STRUCTURED.value == "structured"

    def test_movement_value(self) -> None:
        """Test MOVEMENT value."""
        assert PruningMethod.MOVEMENT.value == "movement"

    def test_lottery_ticket_value(self) -> None:
        """Test LOTTERY_TICKET value."""
        assert PruningMethod.LOTTERY_TICKET.value == "lottery_ticket"

    def test_gradual_value(self) -> None:
        """Test GRADUAL value."""
        assert PruningMethod.GRADUAL.value == "gradual"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [m.value for m in PruningMethod]
        assert len(values) == len(set(values))


class TestPruningScope:
    """Tests for PruningScope enum."""

    def test_global_value(self) -> None:
        """Test GLOBAL value."""
        assert PruningScope.GLOBAL.value == "global"

    def test_local_value(self) -> None:
        """Test LOCAL value."""
        assert PruningScope.LOCAL.value == "local"

    def test_layer_wise_value(self) -> None:
        """Test LAYER_WISE value."""
        assert PruningScope.LAYER_WISE.value == "layer_wise"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [s.value for s in PruningScope]
        assert len(values) == len(set(values))


class TestStructuredPruningDim:
    """Tests for StructuredPruningDim enum."""

    def test_channel_value(self) -> None:
        """Test CHANNEL value."""
        assert StructuredPruningDim.CHANNEL.value == "channel"

    def test_head_value(self) -> None:
        """Test HEAD value."""
        assert StructuredPruningDim.HEAD.value == "head"

    def test_layer_value(self) -> None:
        """Test LAYER value."""
        assert StructuredPruningDim.LAYER.value == "layer"

    def test_block_value(self) -> None:
        """Test BLOCK value."""
        assert StructuredPruningDim.BLOCK.value == "block"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [d.value for d in StructuredPruningDim]
        assert len(values) == len(set(values))


class TestFusionType:
    """Tests for FusionType enum."""

    def test_conv_bn_value(self) -> None:
        """Test CONV_BN value."""
        assert FusionType.CONV_BN.value == "conv_bn"

    def test_linear_bn_value(self) -> None:
        """Test LINEAR_BN value."""
        assert FusionType.LINEAR_BN.value == "linear_bn"

    def test_attention_value(self) -> None:
        """Test ATTENTION value."""
        assert FusionType.ATTENTION.value == "attention"

    def test_mlp_value(self) -> None:
        """Test MLP value."""
        assert FusionType.MLP.value == "mlp"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [f.value for f in FusionType]
        assert len(values) == len(set(values))


class TestDecompositionMethod:
    """Tests for DecompositionMethod enum."""

    def test_svd_value(self) -> None:
        """Test SVD value."""
        assert DecompositionMethod.SVD.value == "svd"

    def test_tucker_value(self) -> None:
        """Test TUCKER value."""
        assert DecompositionMethod.TUCKER.value == "tucker"

    def test_cp_value(self) -> None:
        """Test CP value."""
        assert DecompositionMethod.CP.value == "cp"

    def test_nmf_value(self) -> None:
        """Test NMF value."""
        assert DecompositionMethod.NMF.value == "nmf"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [d.value for d in DecompositionMethod]
        assert len(values) == len(set(values))


class TestImportanceMetric:
    """Tests for ImportanceMetric enum."""

    def test_l1_norm_value(self) -> None:
        """Test L1_NORM value."""
        assert ImportanceMetric.L1_NORM.value == "l1_norm"

    def test_l2_norm_value(self) -> None:
        """Test L2_NORM value."""
        assert ImportanceMetric.L2_NORM.value == "l2_norm"

    def test_gradient_value(self) -> None:
        """Test GRADIENT value."""
        assert ImportanceMetric.GRADIENT.value == "gradient"

    def test_taylor_value(self) -> None:
        """Test TAYLOR value."""
        assert ImportanceMetric.TAYLOR.value == "taylor"

    def test_hessian_value(self) -> None:
        """Test HESSIAN value."""
        assert ImportanceMetric.HESSIAN.value == "hessian"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [m.value for m in ImportanceMetric]
        assert len(values) == len(set(values))


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestPruningSchedule:
    """Tests for PruningSchedule dataclass."""

    def test_creation(self) -> None:
        """Test creating PruningSchedule."""
        schedule = PruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
            frequency=100,
        )
        assert schedule.initial_sparsity == 0.0
        assert schedule.final_sparsity == 0.9
        assert schedule.begin_step == 0
        assert schedule.end_step == 1000
        assert schedule.frequency == 100

    def test_frozen(self) -> None:
        """Test that PruningSchedule is immutable."""
        schedule = PruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
            frequency=100,
        )
        with pytest.raises(AttributeError):
            schedule.final_sparsity = 0.5  # type: ignore[misc]


class TestPruningConfig:
    """Tests for PruningConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PruningConfig."""
        config = PruningConfig(
            method=PruningMethod.MAGNITUDE,
            sparsity=0.5,
            scope=PruningScope.GLOBAL,
            importance_metric=ImportanceMetric.L1_NORM,
            schedule=None,
        )
        assert config.method == PruningMethod.MAGNITUDE
        assert config.sparsity == 0.5
        assert config.scope == PruningScope.GLOBAL
        assert config.importance_metric == ImportanceMetric.L1_NORM
        assert config.schedule is None

    def test_with_schedule(self) -> None:
        """Test PruningConfig with schedule."""
        schedule = PruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
            frequency=100,
        )
        config = PruningConfig(
            method=PruningMethod.GRADUAL,
            sparsity=0.9,
            scope=PruningScope.GLOBAL,
            importance_metric=ImportanceMetric.L1_NORM,
            schedule=schedule,
        )
        assert config.schedule is not None
        assert config.schedule.final_sparsity == 0.9

    def test_frozen(self) -> None:
        """Test that PruningConfig is immutable."""
        config = PruningConfig(
            method=PruningMethod.MAGNITUDE,
            sparsity=0.5,
            scope=PruningScope.GLOBAL,
            importance_metric=ImportanceMetric.L1_NORM,
            schedule=None,
        )
        with pytest.raises(AttributeError):
            config.sparsity = 0.7  # type: ignore[misc]


class TestStructuredPruningConfig:
    """Tests for StructuredPruningConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating StructuredPruningConfig."""
        config = StructuredPruningConfig(
            dimension=StructuredPruningDim.CHANNEL,
            pruning_ratio=0.3,
            importance_metric=ImportanceMetric.L2_NORM,
            min_structures=1,
        )
        assert config.dimension == StructuredPruningDim.CHANNEL
        assert config.pruning_ratio == 0.3
        assert config.importance_metric == ImportanceMetric.L2_NORM
        assert config.min_structures == 1

    def test_frozen(self) -> None:
        """Test that StructuredPruningConfig is immutable."""
        config = StructuredPruningConfig(
            dimension=StructuredPruningDim.CHANNEL,
            pruning_ratio=0.3,
            importance_metric=ImportanceMetric.L2_NORM,
            min_structures=1,
        )
        with pytest.raises(AttributeError):
            config.pruning_ratio = 0.5  # type: ignore[misc]


class TestLayerFusionConfig:
    """Tests for LayerFusionConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating LayerFusionConfig."""
        config = LayerFusionConfig(
            fusion_type=FusionType.CONV_BN,
            fuse_activation=True,
            optimize_for_inference=True,
        )
        assert config.fusion_type == FusionType.CONV_BN
        assert config.fuse_activation is True
        assert config.optimize_for_inference is True

    def test_frozen(self) -> None:
        """Test that LayerFusionConfig is immutable."""
        config = LayerFusionConfig(
            fusion_type=FusionType.CONV_BN,
            fuse_activation=True,
            optimize_for_inference=True,
        )
        with pytest.raises(AttributeError):
            config.fuse_activation = False  # type: ignore[misc]


class TestWeightSharingConfig:
    """Tests for WeightSharingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating WeightSharingConfig."""
        config = WeightSharingConfig(
            num_clusters=256,
            clustering_method="kmeans",
            fine_tune_centroids=True,
            bits_per_weight=8,
        )
        assert config.num_clusters == 256
        assert config.clustering_method == "kmeans"
        assert config.fine_tune_centroids is True
        assert config.bits_per_weight == 8

    def test_frozen(self) -> None:
        """Test that WeightSharingConfig is immutable."""
        config = WeightSharingConfig(
            num_clusters=256,
            clustering_method="kmeans",
            fine_tune_centroids=True,
            bits_per_weight=8,
        )
        with pytest.raises(AttributeError):
            config.num_clusters = 128  # type: ignore[misc]


class TestLowRankConfig:
    """Tests for LowRankConfig dataclass."""

    def test_creation_with_rank(self) -> None:
        """Test creating LowRankConfig with rank."""
        config = LowRankConfig(
            rank=64,
            decomposition_method=DecompositionMethod.SVD,
            rank_ratio=None,
            energy_threshold=0.99,
        )
        assert config.rank == 64
        assert config.decomposition_method == DecompositionMethod.SVD
        assert config.rank_ratio is None
        assert config.energy_threshold == 0.99

    def test_creation_with_rank_ratio(self) -> None:
        """Test creating LowRankConfig with rank_ratio."""
        config = LowRankConfig(
            rank=None,
            decomposition_method=DecompositionMethod.SVD,
            rank_ratio=0.5,
            energy_threshold=0.99,
        )
        assert config.rank is None
        assert config.rank_ratio == 0.5

    def test_frozen(self) -> None:
        """Test that LowRankConfig is immutable."""
        config = LowRankConfig(
            rank=64,
            decomposition_method=DecompositionMethod.SVD,
            rank_ratio=None,
            energy_threshold=0.99,
        )
        with pytest.raises(AttributeError):
            config.rank = 32  # type: ignore[misc]


class TestCompressionConfig:
    """Tests for CompressionConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating CompressionConfig."""
        config = CompressionConfig(
            enable_pruning=True,
            enable_fusion=True,
            enable_weight_sharing=False,
            enable_low_rank=False,
            target_speedup=2.0,
            target_size_reduction=0.5,
        )
        assert config.enable_pruning is True
        assert config.enable_fusion is True
        assert config.enable_weight_sharing is False
        assert config.enable_low_rank is False
        assert config.target_speedup == 2.0
        assert config.target_size_reduction == 0.5

    def test_frozen(self) -> None:
        """Test that CompressionConfig is immutable."""
        config = CompressionConfig(
            enable_pruning=True,
            enable_fusion=True,
            enable_weight_sharing=False,
            enable_low_rank=False,
            target_speedup=2.0,
            target_size_reduction=0.5,
        )
        with pytest.raises(AttributeError):
            config.target_speedup = 3.0  # type: ignore[misc]


class TestCompressionStats:
    """Tests for CompressionStats dataclass."""

    def test_creation(self) -> None:
        """Test creating CompressionStats."""
        stats = CompressionStats(
            original_params=110_000_000,
            compressed_params=55_000_000,
            sparsity=0.5,
            compression_ratio=2.0,
            estimated_speedup=1.8,
            original_size_mb=440.0,
            compressed_size_mb=220.0,
        )
        assert stats.original_params == 110_000_000
        assert stats.compressed_params == 55_000_000
        assert stats.sparsity == 0.5
        assert stats.compression_ratio == 2.0
        assert stats.estimated_speedup == 1.8
        assert stats.original_size_mb == 440.0
        assert stats.compressed_size_mb == 220.0

    def test_frozen(self) -> None:
        """Test that CompressionStats is immutable."""
        stats = CompressionStats(
            original_params=110_000_000,
            compressed_params=55_000_000,
            sparsity=0.5,
            compression_ratio=2.0,
            estimated_speedup=1.8,
            original_size_mb=440.0,
            compressed_size_mb=220.0,
        )
        with pytest.raises(AttributeError):
            stats.sparsity = 0.6  # type: ignore[misc]


# =============================================================================
# Validation Function Tests
# =============================================================================


class TestValidatePruningConfig:
    """Tests for validate_pruning_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            0.5,
            PruningScope.GLOBAL,
            ImportanceMetric.L1_NORM,
            None,
        )
        validate_pruning_config(config)  # Should not raise

    def test_valid_config_with_schedule(self) -> None:
        """Test validating valid config with schedule."""
        schedule = PruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
            frequency=100,
        )
        config = PruningConfig(
            PruningMethod.GRADUAL,
            0.9,
            PruningScope.GLOBAL,
            ImportanceMetric.L1_NORM,
            schedule,
        )
        validate_pruning_config(config)  # Should not raise

    def test_sparsity_too_low_raises_error(self) -> None:
        """Test that sparsity below 0 raises ValueError."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            -0.1,
            PruningScope.GLOBAL,
            ImportanceMetric.L1_NORM,
            None,
        )
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            validate_pruning_config(config)

    def test_sparsity_too_high_raises_error(self) -> None:
        """Test that sparsity above 1 raises ValueError."""
        config = PruningConfig(
            PruningMethod.MAGNITUDE,
            1.5,
            PruningScope.GLOBAL,
            ImportanceMetric.L1_NORM,
            None,
        )
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            validate_pruning_config(config)

    def test_schedule_negative_begin_step_raises_error(self) -> None:
        """Test that negative begin_step raises ValueError."""
        schedule = PruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=-1,
            end_step=1000,
            frequency=100,
        )
        config = PruningConfig(
            PruningMethod.GRADUAL,
            0.9,
            PruningScope.GLOBAL,
            ImportanceMetric.L1_NORM,
            schedule,
        )
        with pytest.raises(ValueError, match="begin_step must be non-negative"):
            validate_pruning_config(config)

    def test_schedule_end_before_begin_raises_error(self) -> None:
        """Test that end_step <= begin_step raises ValueError."""
        schedule = PruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=1000,
            end_step=500,
            frequency=100,
        )
        config = PruningConfig(
            PruningMethod.GRADUAL,
            0.9,
            PruningScope.GLOBAL,
            ImportanceMetric.L1_NORM,
            schedule,
        )
        with pytest.raises(
            ValueError, match="end_step must be greater than begin_step"
        ):
            validate_pruning_config(config)

    def test_schedule_zero_frequency_raises_error(self) -> None:
        """Test that zero frequency raises ValueError."""
        schedule = PruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
            frequency=0,
        )
        config = PruningConfig(
            PruningMethod.GRADUAL,
            0.9,
            PruningScope.GLOBAL,
            ImportanceMetric.L1_NORM,
            schedule,
        )
        with pytest.raises(ValueError, match="frequency must be positive"):
            validate_pruning_config(config)


class TestValidateStructuredPruningConfig:
    """Tests for validate_structured_pruning_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = StructuredPruningConfig(
            StructuredPruningDim.CHANNEL,
            0.3,
            ImportanceMetric.L2_NORM,
            1,
        )
        validate_structured_pruning_config(config)  # Should not raise

    def test_pruning_ratio_too_low_raises_error(self) -> None:
        """Test that pruning_ratio below 0 raises ValueError."""
        config = StructuredPruningConfig(
            StructuredPruningDim.CHANNEL,
            -0.1,
            ImportanceMetric.L2_NORM,
            1,
        )
        with pytest.raises(ValueError, match="pruning_ratio must be between 0 and 1"):
            validate_structured_pruning_config(config)

    def test_pruning_ratio_too_high_raises_error(self) -> None:
        """Test that pruning_ratio above 1 raises ValueError."""
        config = StructuredPruningConfig(
            StructuredPruningDim.CHANNEL,
            1.5,
            ImportanceMetric.L2_NORM,
            1,
        )
        with pytest.raises(ValueError, match="pruning_ratio must be between 0 and 1"):
            validate_structured_pruning_config(config)

    def test_negative_min_structures_raises_error(self) -> None:
        """Test that negative min_structures raises ValueError."""
        config = StructuredPruningConfig(
            StructuredPruningDim.CHANNEL,
            0.3,
            ImportanceMetric.L2_NORM,
            -1,
        )
        with pytest.raises(ValueError, match="min_structures must be non-negative"):
            validate_structured_pruning_config(config)


class TestValidateWeightSharingConfig:
    """Tests for validate_weight_sharing_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = WeightSharingConfig(256, "kmeans", True, 8)
        validate_weight_sharing_config(config)  # Should not raise

    def test_zero_clusters_raises_error(self) -> None:
        """Test that zero num_clusters raises ValueError."""
        config = WeightSharingConfig(0, "kmeans", True, 8)
        with pytest.raises(ValueError, match="num_clusters must be positive"):
            validate_weight_sharing_config(config)

    def test_negative_clusters_raises_error(self) -> None:
        """Test that negative num_clusters raises ValueError."""
        config = WeightSharingConfig(-1, "kmeans", True, 8)
        with pytest.raises(ValueError, match="num_clusters must be positive"):
            validate_weight_sharing_config(config)

    def test_zero_bits_raises_error(self) -> None:
        """Test that zero bits_per_weight raises ValueError."""
        config = WeightSharingConfig(256, "kmeans", True, 0)
        with pytest.raises(ValueError, match="bits_per_weight must be positive"):
            validate_weight_sharing_config(config)

    def test_clusters_exceed_bits_raises_error(self) -> None:
        """Test that num_clusters exceeding 2^bits raises ValueError."""
        config = WeightSharingConfig(300, "kmeans", True, 8)  # 300 > 256 = 2^8
        with pytest.raises(ValueError, match=r"num_clusters.*exceeds maximum"):
            validate_weight_sharing_config(config)


class TestValidateLowRankConfig:
    """Tests for validate_low_rank_config function."""

    def test_valid_config_with_rank(self) -> None:
        """Test validating valid config with rank."""
        config = LowRankConfig(64, DecompositionMethod.SVD, None, 0.99)
        validate_low_rank_config(config)  # Should not raise

    def test_valid_config_with_rank_ratio(self) -> None:
        """Test validating valid config with rank_ratio."""
        config = LowRankConfig(None, DecompositionMethod.SVD, 0.5, 0.99)
        validate_low_rank_config(config)  # Should not raise

    def test_zero_rank_raises_error(self) -> None:
        """Test that zero rank raises ValueError."""
        config = LowRankConfig(0, DecompositionMethod.SVD, None, 0.99)
        with pytest.raises(ValueError, match="rank must be positive"):
            validate_low_rank_config(config)

    def test_negative_rank_raises_error(self) -> None:
        """Test that negative rank raises ValueError."""
        config = LowRankConfig(-1, DecompositionMethod.SVD, None, 0.99)
        with pytest.raises(ValueError, match="rank must be positive"):
            validate_low_rank_config(config)

    def test_rank_ratio_zero_raises_error(self) -> None:
        """Test that rank_ratio of 0 raises ValueError."""
        config = LowRankConfig(None, DecompositionMethod.SVD, 0.0, 0.99)
        with pytest.raises(ValueError, match="rank_ratio must be between 0 and 1"):
            validate_low_rank_config(config)

    def test_rank_ratio_above_one_raises_error(self) -> None:
        """Test that rank_ratio above 1 raises ValueError."""
        config = LowRankConfig(None, DecompositionMethod.SVD, 1.5, 0.99)
        with pytest.raises(ValueError, match="rank_ratio must be between 0 and 1"):
            validate_low_rank_config(config)

    def test_energy_threshold_zero_raises_error(self) -> None:
        """Test that energy_threshold of 0 raises ValueError."""
        config = LowRankConfig(64, DecompositionMethod.SVD, None, 0.0)
        with pytest.raises(
            ValueError, match="energy_threshold must be between 0 and 1"
        ):
            validate_low_rank_config(config)

    def test_energy_threshold_above_one_raises_error(self) -> None:
        """Test that energy_threshold above 1 raises ValueError."""
        config = LowRankConfig(64, DecompositionMethod.SVD, None, 1.5)
        with pytest.raises(
            ValueError, match="energy_threshold must be between 0 and 1"
        ):
            validate_low_rank_config(config)

    def test_neither_rank_nor_ratio_raises_error(self) -> None:
        """Test that neither rank nor rank_ratio raises ValueError."""
        config = LowRankConfig(None, DecompositionMethod.SVD, None, 0.99)
        with pytest.raises(
            ValueError, match="either rank or rank_ratio must be specified"
        ):
            validate_low_rank_config(config)


class TestValidateCompressionConfig:
    """Tests for validate_compression_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = CompressionConfig(True, True, False, False, 2.0, 0.5)
        validate_compression_config(config)  # Should not raise

    def test_speedup_below_one_raises_error(self) -> None:
        """Test that target_speedup below 1.0 raises ValueError."""
        config = CompressionConfig(True, True, False, False, 0.5, 0.5)
        with pytest.raises(ValueError, match=r"target_speedup must be >= 1\.0"):
            validate_compression_config(config)

    def test_size_reduction_zero_raises_error(self) -> None:
        """Test that target_size_reduction of 0 raises ValueError."""
        config = CompressionConfig(True, True, False, False, 2.0, 0.0)
        with pytest.raises(
            ValueError, match="target_size_reduction must be between 0 and 1"
        ):
            validate_compression_config(config)

    def test_size_reduction_above_one_raises_error(self) -> None:
        """Test that target_size_reduction above 1 raises ValueError."""
        config = CompressionConfig(True, True, False, False, 2.0, 1.5)
        with pytest.raises(
            ValueError, match="target_size_reduction must be between 0 and 1"
        ):
            validate_compression_config(config)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreatePruningSchedule:
    """Tests for create_pruning_schedule function."""

    def test_default_values(self) -> None:
        """Test creating schedule with default values."""
        schedule = create_pruning_schedule()
        assert schedule.initial_sparsity == 0.0
        assert schedule.final_sparsity == 0.9
        assert schedule.begin_step == 0
        assert schedule.end_step == 1000
        assert schedule.frequency == 100

    def test_custom_values(self) -> None:
        """Test creating schedule with custom values."""
        schedule = create_pruning_schedule(
            initial_sparsity=0.1,
            final_sparsity=0.8,
            begin_step=100,
            end_step=2000,
            frequency=50,
        )
        assert schedule.initial_sparsity == 0.1
        assert schedule.final_sparsity == 0.8
        assert schedule.begin_step == 100
        assert schedule.end_step == 2000
        assert schedule.frequency == 50

    def test_initial_sparsity_too_low_raises_error(self) -> None:
        """Test that initial_sparsity below 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="initial_sparsity must be between 0 and 1"
        ):
            create_pruning_schedule(initial_sparsity=-0.1)

    def test_initial_sparsity_too_high_raises_error(self) -> None:
        """Test that initial_sparsity above 1 raises ValueError."""
        with pytest.raises(
            ValueError, match="initial_sparsity must be between 0 and 1"
        ):
            create_pruning_schedule(initial_sparsity=1.5)

    def test_final_sparsity_too_low_raises_error(self) -> None:
        """Test that final_sparsity below 0 raises ValueError."""
        with pytest.raises(ValueError, match="final_sparsity must be between 0 and 1"):
            create_pruning_schedule(final_sparsity=-0.1)

    def test_final_sparsity_too_high_raises_error(self) -> None:
        """Test that final_sparsity above 1 raises ValueError."""
        with pytest.raises(ValueError, match="final_sparsity must be between 0 and 1"):
            create_pruning_schedule(final_sparsity=1.5)

    def test_negative_begin_step_raises_error(self) -> None:
        """Test that negative begin_step raises ValueError."""
        with pytest.raises(ValueError, match="begin_step must be non-negative"):
            create_pruning_schedule(begin_step=-1)

    def test_end_before_begin_raises_error(self) -> None:
        """Test that end_step <= begin_step raises ValueError."""
        with pytest.raises(
            ValueError, match="end_step must be greater than begin_step"
        ):
            create_pruning_schedule(begin_step=1000, end_step=500)

    def test_zero_frequency_raises_error(self) -> None:
        """Test that zero frequency raises ValueError."""
        with pytest.raises(ValueError, match="frequency must be positive"):
            create_pruning_schedule(frequency=0)

    def test_negative_frequency_raises_error(self) -> None:
        """Test that negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="frequency must be positive"):
            create_pruning_schedule(frequency=-1)


class TestCreatePruningConfig:
    """Tests for create_pruning_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_pruning_config()
        assert config.method == PruningMethod.MAGNITUDE
        assert config.sparsity == 0.5
        assert config.scope == PruningScope.GLOBAL
        assert config.importance_metric == ImportanceMetric.L1_NORM
        assert config.schedule is None

    def test_string_method(self) -> None:
        """Test creating config with string method."""
        config = create_pruning_config(method="structured")
        assert config.method == PruningMethod.STRUCTURED

    def test_enum_method(self) -> None:
        """Test creating config with enum method."""
        config = create_pruning_config(method=PruningMethod.MOVEMENT)
        assert config.method == PruningMethod.MOVEMENT

    def test_string_scope(self) -> None:
        """Test creating config with string scope."""
        config = create_pruning_config(scope="local")
        assert config.scope == PruningScope.LOCAL

    def test_string_importance_metric(self) -> None:
        """Test creating config with string importance_metric."""
        config = create_pruning_config(importance_metric="gradient")
        assert config.importance_metric == ImportanceMetric.GRADIENT

    def test_with_schedule(self) -> None:
        """Test creating config with schedule."""
        schedule = create_pruning_schedule()
        config = create_pruning_config(schedule=schedule)
        assert config.schedule is not None
        assert config.schedule.final_sparsity == 0.9

    def test_invalid_sparsity_raises_error(self) -> None:
        """Test that invalid sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            create_pruning_config(sparsity=1.5)

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_pruning_config(method="invalid")


class TestCreateStructuredPruningConfig:
    """Tests for create_structured_pruning_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_structured_pruning_config()
        assert config.dimension == StructuredPruningDim.CHANNEL
        assert config.pruning_ratio == 0.3
        assert config.importance_metric == ImportanceMetric.L2_NORM
        assert config.min_structures == 1

    def test_string_dimension(self) -> None:
        """Test creating config with string dimension."""
        config = create_structured_pruning_config(dimension="head")
        assert config.dimension == StructuredPruningDim.HEAD

    def test_enum_dimension(self) -> None:
        """Test creating config with enum dimension."""
        config = create_structured_pruning_config(dimension=StructuredPruningDim.LAYER)
        assert config.dimension == StructuredPruningDim.LAYER

    def test_string_importance_metric(self) -> None:
        """Test creating config with string importance_metric."""
        config = create_structured_pruning_config(importance_metric="taylor")
        assert config.importance_metric == ImportanceMetric.TAYLOR

    def test_invalid_pruning_ratio_raises_error(self) -> None:
        """Test that invalid pruning_ratio raises ValueError."""
        with pytest.raises(ValueError, match="pruning_ratio must be between 0 and 1"):
            create_structured_pruning_config(pruning_ratio=1.5)

    def test_invalid_dimension_raises_error(self) -> None:
        """Test that invalid dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be one of"):
            create_structured_pruning_config(dimension="invalid")


class TestCreateLayerFusionConfig:
    """Tests for create_layer_fusion_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_layer_fusion_config()
        assert config.fusion_type == FusionType.CONV_BN
        assert config.fuse_activation is True
        assert config.optimize_for_inference is True

    def test_string_fusion_type(self) -> None:
        """Test creating config with string fusion_type."""
        config = create_layer_fusion_config(fusion_type="linear_bn")
        assert config.fusion_type == FusionType.LINEAR_BN

    def test_enum_fusion_type(self) -> None:
        """Test creating config with enum fusion_type."""
        config = create_layer_fusion_config(fusion_type=FusionType.ATTENTION)
        assert config.fusion_type == FusionType.ATTENTION

    def test_custom_flags(self) -> None:
        """Test creating config with custom flags."""
        config = create_layer_fusion_config(
            fuse_activation=False,
            optimize_for_inference=False,
        )
        assert config.fuse_activation is False
        assert config.optimize_for_inference is False

    def test_invalid_fusion_type_raises_error(self) -> None:
        """Test that invalid fusion_type raises ValueError."""
        with pytest.raises(ValueError, match="fusion_type must be one of"):
            create_layer_fusion_config(fusion_type="invalid")


class TestCreateWeightSharingConfig:
    """Tests for create_weight_sharing_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_weight_sharing_config()
        assert config.num_clusters == 256
        assert config.clustering_method == "kmeans"
        assert config.fine_tune_centroids is True
        assert config.bits_per_weight == 8

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_weight_sharing_config(
            num_clusters=16,
            clustering_method="linear",
            fine_tune_centroids=False,
            bits_per_weight=4,
        )
        assert config.num_clusters == 16
        assert config.clustering_method == "linear"
        assert config.fine_tune_centroids is False
        assert config.bits_per_weight == 4

    def test_uniform_clustering_method(self) -> None:
        """Test creating config with uniform clustering method."""
        config = create_weight_sharing_config(clustering_method="uniform")
        assert config.clustering_method == "uniform"

    def test_zero_clusters_raises_error(self) -> None:
        """Test that zero num_clusters raises ValueError."""
        with pytest.raises(ValueError, match="num_clusters must be positive"):
            create_weight_sharing_config(num_clusters=0)

    def test_invalid_clustering_method_raises_error(self) -> None:
        """Test that invalid clustering_method raises ValueError."""
        with pytest.raises(ValueError, match="clustering_method must be one of"):
            create_weight_sharing_config(clustering_method="invalid")

    def test_clusters_exceed_bits_raises_error(self) -> None:
        """Test that num_clusters exceeding 2^bits raises ValueError."""
        with pytest.raises(ValueError, match=r"num_clusters.*exceeds maximum"):
            create_weight_sharing_config(num_clusters=300, bits_per_weight=8)


class TestCreateLowRankConfig:
    """Tests for create_low_rank_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_low_rank_config()
        assert config.rank == 64
        assert config.decomposition_method == DecompositionMethod.SVD
        assert config.rank_ratio is None
        assert config.energy_threshold == 0.99

    def test_string_decomposition_method(self) -> None:
        """Test creating config with string decomposition_method."""
        config = create_low_rank_config(decomposition_method="tucker")
        assert config.decomposition_method == DecompositionMethod.TUCKER

    def test_enum_decomposition_method(self) -> None:
        """Test creating config with enum decomposition_method."""
        config = create_low_rank_config(decomposition_method=DecompositionMethod.NMF)
        assert config.decomposition_method == DecompositionMethod.NMF

    def test_with_rank_ratio(self) -> None:
        """Test creating config with rank_ratio."""
        config = create_low_rank_config(rank=None, rank_ratio=0.5)
        assert config.rank is None
        assert config.rank_ratio == 0.5

    def test_zero_rank_raises_error(self) -> None:
        """Test that zero rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            create_low_rank_config(rank=0)

    def test_invalid_decomposition_method_raises_error(self) -> None:
        """Test that invalid decomposition_method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_low_rank_config(decomposition_method="invalid")


class TestCreateCompressionConfig:
    """Tests for create_compression_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_compression_config()
        assert config.enable_pruning is True
        assert config.enable_fusion is True
        assert config.enable_weight_sharing is False
        assert config.enable_low_rank is False
        assert config.target_speedup == 2.0
        assert config.target_size_reduction == 0.5

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_compression_config(
            enable_pruning=False,
            enable_fusion=False,
            enable_weight_sharing=True,
            enable_low_rank=True,
            target_speedup=4.0,
            target_size_reduction=0.25,
        )
        assert config.enable_pruning is False
        assert config.enable_fusion is False
        assert config.enable_weight_sharing is True
        assert config.enable_low_rank is True
        assert config.target_speedup == 4.0
        assert config.target_size_reduction == 0.25

    def test_speedup_below_one_raises_error(self) -> None:
        """Test that target_speedup below 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"target_speedup must be >= 1\.0"):
            create_compression_config(target_speedup=0.5)

    def test_size_reduction_zero_raises_error(self) -> None:
        """Test that target_size_reduction of 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="target_size_reduction must be between 0 and 1"
        ):
            create_compression_config(target_size_reduction=0.0)

    def test_size_reduction_above_one_raises_error(self) -> None:
        """Test that target_size_reduction above 1 raises ValueError."""
        with pytest.raises(
            ValueError, match="target_size_reduction must be between 0 and 1"
        ):
            create_compression_config(target_size_reduction=1.5)


class TestCreateCompressionStats:
    """Tests for create_compression_stats function."""

    def test_creates_stats(self) -> None:
        """Test creating stats."""
        stats = create_compression_stats(100_000_000, 50_000_000)
        assert stats.original_params == 100_000_000
        assert stats.compressed_params == 50_000_000
        assert stats.sparsity == pytest.approx(0.5)
        assert stats.compression_ratio == pytest.approx(2.0)

    def test_calculates_sizes(self) -> None:
        """Test that sizes are calculated when not provided."""
        stats = create_compression_stats(100_000_000, 50_000_000)
        # 100M params * 4 bytes / (1024 * 1024) = ~381.47 MB
        expected_original = 100_000_000 * 4 / (1024 * 1024)
        expected_compressed = 50_000_000 * 4 / (1024 * 1024)
        assert stats.original_size_mb == pytest.approx(expected_original)
        assert stats.compressed_size_mb == pytest.approx(expected_compressed)

    def test_uses_provided_sizes(self) -> None:
        """Test that provided sizes are used."""
        stats = create_compression_stats(
            100_000_000,
            50_000_000,
            original_size_mb=400.0,
            compressed_size_mb=200.0,
        )
        assert stats.original_size_mb == 400.0
        assert stats.compressed_size_mb == 200.0

    def test_zero_compressed_params(self) -> None:
        """Test stats with zero compressed params."""
        stats = create_compression_stats(100_000_000, 0)
        assert stats.sparsity == pytest.approx(1.0)
        assert math.isinf(stats.compression_ratio)

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original_params raises ValueError."""
        with pytest.raises(ValueError, match="original_params must be positive"):
            create_compression_stats(0, 50_000_000)

    def test_negative_original_raises_error(self) -> None:
        """Test that negative original_params raises ValueError."""
        with pytest.raises(ValueError, match="original_params must be positive"):
            create_compression_stats(-1, 50_000_000)

    def test_negative_compressed_raises_error(self) -> None:
        """Test that negative compressed_params raises ValueError."""
        with pytest.raises(ValueError, match="compressed_params must be non-negative"):
            create_compression_stats(100_000_000, -1)

    def test_compressed_exceeds_original_raises_error(self) -> None:
        """Test that compressed_params > original_params raises ValueError."""
        with pytest.raises(ValueError, match=r"compressed_params.*cannot exceed"):
            create_compression_stats(50_000_000, 100_000_000)


# =============================================================================
# List Function Tests
# =============================================================================


class TestListPruningMethods:
    """Tests for list_pruning_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_pruning_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_pruning_methods()
        assert "magnitude" in methods
        assert "structured" in methods
        assert "movement" in methods
        assert "lottery_ticket" in methods
        assert "gradual" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_pruning_methods()
        assert methods == sorted(methods)


class TestListPruningScopes:
    """Tests for list_pruning_scopes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        scopes = list_pruning_scopes()
        assert isinstance(scopes, list)

    def test_contains_expected_scopes(self) -> None:
        """Test that list contains expected scopes."""
        scopes = list_pruning_scopes()
        assert "global" in scopes
        assert "local" in scopes
        assert "layer_wise" in scopes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        scopes = list_pruning_scopes()
        assert scopes == sorted(scopes)


class TestListStructuredPruningDims:
    """Tests for list_structured_pruning_dims function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        dims = list_structured_pruning_dims()
        assert isinstance(dims, list)

    def test_contains_expected_dims(self) -> None:
        """Test that list contains expected dimensions."""
        dims = list_structured_pruning_dims()
        assert "channel" in dims
        assert "head" in dims
        assert "layer" in dims
        assert "block" in dims

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        dims = list_structured_pruning_dims()
        assert dims == sorted(dims)


class TestListFusionTypes:
    """Tests for list_fusion_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_fusion_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_fusion_types()
        assert "conv_bn" in types
        assert "linear_bn" in types
        assert "attention" in types
        assert "mlp" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_fusion_types()
        assert types == sorted(types)


class TestListDecompositionMethods:
    """Tests for list_decomposition_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_decomposition_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_decomposition_methods()
        assert "svd" in methods
        assert "tucker" in methods
        assert "cp" in methods
        assert "nmf" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_decomposition_methods()
        assert methods == sorted(methods)


class TestListImportanceMetrics:
    """Tests for list_importance_metrics function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        metrics = list_importance_metrics()
        assert isinstance(metrics, list)

    def test_contains_expected_metrics(self) -> None:
        """Test that list contains expected metrics."""
        metrics = list_importance_metrics()
        assert "l1_norm" in metrics
        assert "l2_norm" in metrics
        assert "gradient" in metrics
        assert "taylor" in metrics
        assert "hessian" in metrics

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        metrics = list_importance_metrics()
        assert metrics == sorted(metrics)


# =============================================================================
# Get Function Tests
# =============================================================================


class TestGetPruningMethod:
    """Tests for get_pruning_method function."""

    def test_get_magnitude(self) -> None:
        """Test getting MAGNITUDE method."""
        assert get_pruning_method("magnitude") == PruningMethod.MAGNITUDE

    def test_get_structured(self) -> None:
        """Test getting STRUCTURED method."""
        assert get_pruning_method("structured") == PruningMethod.STRUCTURED

    def test_get_movement(self) -> None:
        """Test getting MOVEMENT method."""
        assert get_pruning_method("movement") == PruningMethod.MOVEMENT

    def test_get_lottery_ticket(self) -> None:
        """Test getting LOTTERY_TICKET method."""
        assert get_pruning_method("lottery_ticket") == PruningMethod.LOTTERY_TICKET

    def test_get_gradual(self) -> None:
        """Test getting GRADUAL method."""
        assert get_pruning_method("gradual") == PruningMethod.GRADUAL

    def test_invalid_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_pruning_method("invalid")


class TestGetPruningScope:
    """Tests for get_pruning_scope function."""

    def test_get_global(self) -> None:
        """Test getting GLOBAL scope."""
        assert get_pruning_scope("global") == PruningScope.GLOBAL

    def test_get_local(self) -> None:
        """Test getting LOCAL scope."""
        assert get_pruning_scope("local") == PruningScope.LOCAL

    def test_get_layer_wise(self) -> None:
        """Test getting LAYER_WISE scope."""
        assert get_pruning_scope("layer_wise") == PruningScope.LAYER_WISE

    def test_invalid_raises_error(self) -> None:
        """Test that invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="scope must be one of"):
            get_pruning_scope("invalid")


class TestGetStructuredPruningDim:
    """Tests for get_structured_pruning_dim function."""

    def test_get_channel(self) -> None:
        """Test getting CHANNEL dimension."""
        assert get_structured_pruning_dim("channel") == StructuredPruningDim.CHANNEL

    def test_get_head(self) -> None:
        """Test getting HEAD dimension."""
        assert get_structured_pruning_dim("head") == StructuredPruningDim.HEAD

    def test_get_layer(self) -> None:
        """Test getting LAYER dimension."""
        assert get_structured_pruning_dim("layer") == StructuredPruningDim.LAYER

    def test_get_block(self) -> None:
        """Test getting BLOCK dimension."""
        assert get_structured_pruning_dim("block") == StructuredPruningDim.BLOCK

    def test_invalid_raises_error(self) -> None:
        """Test that invalid dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be one of"):
            get_structured_pruning_dim("invalid")


class TestGetFusionType:
    """Tests for get_fusion_type function."""

    def test_get_conv_bn(self) -> None:
        """Test getting CONV_BN type."""
        assert get_fusion_type("conv_bn") == FusionType.CONV_BN

    def test_get_linear_bn(self) -> None:
        """Test getting LINEAR_BN type."""
        assert get_fusion_type("linear_bn") == FusionType.LINEAR_BN

    def test_get_attention(self) -> None:
        """Test getting ATTENTION type."""
        assert get_fusion_type("attention") == FusionType.ATTENTION

    def test_get_mlp(self) -> None:
        """Test getting MLP type."""
        assert get_fusion_type("mlp") == FusionType.MLP

    def test_invalid_raises_error(self) -> None:
        """Test that invalid fusion type raises ValueError."""
        with pytest.raises(ValueError, match="fusion_type must be one of"):
            get_fusion_type("invalid")


class TestGetDecompositionMethod:
    """Tests for get_decomposition_method function."""

    def test_get_svd(self) -> None:
        """Test getting SVD method."""
        assert get_decomposition_method("svd") == DecompositionMethod.SVD

    def test_get_tucker(self) -> None:
        """Test getting TUCKER method."""
        assert get_decomposition_method("tucker") == DecompositionMethod.TUCKER

    def test_get_cp(self) -> None:
        """Test getting CP method."""
        assert get_decomposition_method("cp") == DecompositionMethod.CP

    def test_get_nmf(self) -> None:
        """Test getting NMF method."""
        assert get_decomposition_method("nmf") == DecompositionMethod.NMF

    def test_invalid_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_decomposition_method("invalid")


class TestGetImportanceMetric:
    """Tests for get_importance_metric function."""

    def test_get_l1_norm(self) -> None:
        """Test getting L1_NORM metric."""
        assert get_importance_metric("l1_norm") == ImportanceMetric.L1_NORM

    def test_get_l2_norm(self) -> None:
        """Test getting L2_NORM metric."""
        assert get_importance_metric("l2_norm") == ImportanceMetric.L2_NORM

    def test_get_gradient(self) -> None:
        """Test getting GRADIENT metric."""
        assert get_importance_metric("gradient") == ImportanceMetric.GRADIENT

    def test_get_taylor(self) -> None:
        """Test getting TAYLOR metric."""
        assert get_importance_metric("taylor") == ImportanceMetric.TAYLOR

    def test_get_hessian(self) -> None:
        """Test getting HESSIAN metric."""
        assert get_importance_metric("hessian") == ImportanceMetric.HESSIAN

    def test_invalid_raises_error(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            get_importance_metric("invalid")


# =============================================================================
# Calculation Function Tests
# =============================================================================


class TestCalculateSparsityAtStep:
    """Tests for calculate_sparsity_at_step function."""

    def test_at_begin_step(self) -> None:
        """Test sparsity at begin_step."""
        schedule = create_pruning_schedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
        )
        assert calculate_sparsity_at_step(schedule, 0) == 0.0

    def test_at_end_step(self) -> None:
        """Test sparsity at end_step."""
        schedule = create_pruning_schedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
        )
        assert calculate_sparsity_at_step(schedule, 1000) == 0.9

    def test_at_midpoint(self) -> None:
        """Test sparsity at midpoint."""
        schedule = create_pruning_schedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
        )
        assert calculate_sparsity_at_step(schedule, 500) == pytest.approx(0.45)

    def test_before_begin_step(self) -> None:
        """Test sparsity before begin_step returns initial."""
        schedule = create_pruning_schedule(
            initial_sparsity=0.1,
            final_sparsity=0.9,
            begin_step=100,
            end_step=1000,
        )
        assert calculate_sparsity_at_step(schedule, 50) == 0.1

    def test_after_end_step(self) -> None:
        """Test sparsity after end_step returns final."""
        schedule = create_pruning_schedule(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
        )
        assert calculate_sparsity_at_step(schedule, 1500) == 0.9

    def test_linear_interpolation(self) -> None:
        """Test that sparsity interpolates linearly."""
        schedule = create_pruning_schedule(
            initial_sparsity=0.0,
            final_sparsity=1.0,
            begin_step=0,
            end_step=100,
        )
        for step in range(0, 101, 10):
            expected = step / 100.0
            assert calculate_sparsity_at_step(schedule, step) == pytest.approx(expected)


class TestEstimateSpeedupFromSparsity:
    """Tests for estimate_speedup_from_sparsity function."""

    def test_zero_sparsity(self) -> None:
        """Test speedup with zero sparsity."""
        assert estimate_speedup_from_sparsity(0.0) == pytest.approx(1.0)

    def test_half_sparsity(self) -> None:
        """Test speedup with 50% sparsity."""
        assert estimate_speedup_from_sparsity(0.5) == pytest.approx(2.0)

    def test_ninety_percent_sparsity(self) -> None:
        """Test speedup with 90% sparsity."""
        assert estimate_speedup_from_sparsity(0.9) == pytest.approx(10.0)

    def test_full_sparsity(self) -> None:
        """Test speedup with 100% sparsity returns infinity."""
        assert math.isinf(estimate_speedup_from_sparsity(1.0))

    def test_negative_sparsity_raises_error(self) -> None:
        """Test that negative sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            estimate_speedup_from_sparsity(-0.1)

    def test_sparsity_above_one_raises_error(self) -> None:
        """Test that sparsity above 1 raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            estimate_speedup_from_sparsity(1.5)


class TestEstimateCompressedSize:
    """Tests for estimate_compressed_size function."""

    def test_half_sparsity(self) -> None:
        """Test size with 50% sparsity."""
        size = estimate_compressed_size(100_000_000, 0.5)
        assert size == 200_000_000  # 50M params * 4 bytes

    def test_ninety_percent_sparsity(self) -> None:
        """Test size with 90% sparsity."""
        size = estimate_compressed_size(100_000_000, 0.9)
        # 10M params * 4 bytes = 40M bytes, but int conversion might affect
        assert size == pytest.approx(39_999_996, rel=0.01)

    def test_zero_sparsity(self) -> None:
        """Test size with zero sparsity."""
        size = estimate_compressed_size(100_000_000, 0.0)
        assert size == 400_000_000  # 100M params * 4 bytes

    def test_custom_bits(self) -> None:
        """Test size with custom bits_per_param."""
        size = estimate_compressed_size(100_000_000, 0.5, bits_per_param=16)
        assert size == 100_000_000  # 50M params * 2 bytes

    def test_zero_params_raises_error(self) -> None:
        """Test that zero original_params raises ValueError."""
        with pytest.raises(ValueError, match="original_params must be positive"):
            estimate_compressed_size(0, 0.5)

    def test_negative_params_raises_error(self) -> None:
        """Test that negative original_params raises ValueError."""
        with pytest.raises(ValueError, match="original_params must be positive"):
            estimate_compressed_size(-1, 0.5)

    def test_invalid_sparsity_raises_error(self) -> None:
        """Test that invalid sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            estimate_compressed_size(100_000_000, 1.5)

    def test_zero_bits_raises_error(self) -> None:
        """Test that zero bits_per_param raises ValueError."""
        with pytest.raises(ValueError, match="bits_per_param must be positive"):
            estimate_compressed_size(100_000_000, 0.5, bits_per_param=0)


class TestCalculateLowRankParams:
    """Tests for calculate_low_rank_params function."""

    def test_square_matrix(self) -> None:
        """Test with square matrix."""
        # (768, 768) with rank 64: 768*64 + 64*768 = 98304
        params = calculate_low_rank_params((768, 768), 64)
        assert params == 98304

    def test_rectangular_matrix(self) -> None:
        """Test with rectangular matrix."""
        # (1024, 512) with rank 32: 1024*32 + 32*512 = 49152
        params = calculate_low_rank_params((1024, 512), 32)
        assert params == 49152

    def test_rank_one(self) -> None:
        """Test with rank 1."""
        # (100, 200) with rank 1: 100*1 + 1*200 = 300
        params = calculate_low_rank_params((100, 200), 1)
        assert params == 300

    def test_zero_rank_raises_error(self) -> None:
        """Test that zero rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            calculate_low_rank_params((768, 768), 0)

    def test_negative_rank_raises_error(self) -> None:
        """Test that negative rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            calculate_low_rank_params((768, 768), -1)

    def test_rank_exceeds_min_dim_raises_error(self) -> None:
        """Test that rank exceeding min dimension raises ValueError."""
        with pytest.raises(ValueError, match=r"rank.*cannot exceed min dimension"):
            calculate_low_rank_params((100, 50), 60)

    def test_non_2d_shape_raises_error(self) -> None:
        """Test that non-2D shape raises ValueError."""
        with pytest.raises(ValueError, match="original_shape must be 2D"):
            calculate_low_rank_params((768, 768, 768), 64)  # type: ignore[arg-type]


class TestCalculateWeightSharingBits:
    """Tests for calculate_weight_sharing_bits function."""

    def test_standard_8bit(self) -> None:
        """Test with standard 8-bit clustering."""
        # 100K params * 8 bits + 256 centroids * 32 bits = 808192
        bits = calculate_weight_sharing_bits(100_000, 256, 8)
        assert bits == 808192

    def test_4bit_clustering(self) -> None:
        """Test with 4-bit clustering."""
        # 100K params * 4 bits + 16 centroids * 32 bits = 400512
        bits = calculate_weight_sharing_bits(100_000, 16, 4)
        assert bits == 400512

    def test_minimal_clustering(self) -> None:
        """Test with minimal clustering."""
        # 1000 params * 2 bits + 4 centroids * 32 bits = 2128
        bits = calculate_weight_sharing_bits(1000, 4, 2)
        assert bits == 2128


class TestCalculateFlopsReduction:
    """Tests for calculate_flops_reduction function."""

    def test_half_sparsity_unstructured(self) -> None:
        """Test with 50% sparsity, unstructured."""
        flops = calculate_flops_reduction(1_000_000, 0.5, structured=False)
        assert flops == 500_000

    def test_half_sparsity_structured(self) -> None:
        """Test with 50% sparsity, structured."""
        flops = calculate_flops_reduction(1_000_000, 0.5, structured=True)
        assert flops == 500_000

    def test_high_sparsity_unstructured(self) -> None:
        """Test with high sparsity, unstructured (has minimum factor)."""
        flops = calculate_flops_reduction(1_000_000, 0.95, structured=False)
        # With 95% sparsity, base_reduction = 0.05, but minimum is 0.1
        assert flops == 100_000

    def test_high_sparsity_structured(self) -> None:
        """Test with high sparsity, structured (no minimum factor)."""
        flops = calculate_flops_reduction(1_000_000, 0.95, structured=True)
        # With 95% sparsity, base_reduction = 0.05
        assert flops == 50_000

    def test_zero_sparsity(self) -> None:
        """Test with zero sparsity."""
        flops = calculate_flops_reduction(1_000_000, 0.0)
        assert flops == 1_000_000

    def test_zero_flops_raises_error(self) -> None:
        """Test that zero original_flops raises ValueError."""
        with pytest.raises(ValueError, match="original_flops must be positive"):
            calculate_flops_reduction(0, 0.5)

    def test_negative_flops_raises_error(self) -> None:
        """Test that negative original_flops raises ValueError."""
        with pytest.raises(ValueError, match="original_flops must be positive"):
            calculate_flops_reduction(-1, 0.5)

    def test_invalid_sparsity_raises_error(self) -> None:
        """Test that invalid sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            calculate_flops_reduction(1_000_000, 1.5)


# =============================================================================
# Other Function Tests
# =============================================================================


class TestGetRecommendedCompressionConfig:
    """Tests for get_recommended_compression_config function."""

    def test_inference_config(self) -> None:
        """Test inference task configuration."""
        config = get_recommended_compression_config("inference")
        assert config.enable_pruning is True
        assert config.enable_fusion is True
        assert config.enable_weight_sharing is False
        assert config.enable_low_rank is False
        assert config.target_speedup == 2.0
        assert config.target_size_reduction == 0.5

    def test_mobile_config(self) -> None:
        """Test mobile task configuration."""
        config = get_recommended_compression_config("mobile")
        assert config.enable_pruning is True
        assert config.enable_fusion is True
        assert config.enable_weight_sharing is True
        assert config.enable_low_rank is True
        assert config.target_speedup == 4.0
        assert config.target_size_reduction == 0.25

    def test_edge_config(self) -> None:
        """Test edge task configuration."""
        config = get_recommended_compression_config("edge")
        assert config.enable_pruning is True
        assert config.enable_fusion is True
        assert config.enable_weight_sharing is True
        assert config.enable_low_rank is True
        assert config.target_speedup == 8.0
        assert config.target_size_reduction == 0.1

    def test_research_config(self) -> None:
        """Test research task configuration."""
        config = get_recommended_compression_config("research")
        assert config.enable_pruning is True
        assert config.enable_fusion is False
        assert config.enable_weight_sharing is False
        assert config.enable_low_rank is False
        assert config.target_speedup == 1.5
        assert config.target_size_reduction == 0.7

    def test_invalid_task_raises_error(self) -> None:
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_compression_config("unknown")


class TestFormatCompressionStats:
    """Tests for format_compression_stats function."""

    def test_formats_stats(self) -> None:
        """Test formatting stats."""
        stats = create_compression_stats(100_000_000, 50_000_000)
        formatted = format_compression_stats(stats)
        assert "Compression Stats:" in formatted
        assert "Original Params:" in formatted
        assert "Compressed Params:" in formatted
        assert "Sparsity: 50.0%" in formatted
        assert "Compression Ratio: 2.00x" in formatted
        assert "Est. Speedup:" in formatted
        assert "Original Size:" in formatted
        assert "Compressed Size:" in formatted

    def test_formats_large_numbers(self) -> None:
        """Test formatting with large numbers."""
        stats = create_compression_stats(110_000_000, 55_000_000)
        formatted = format_compression_stats(stats)
        assert "110,000,000" in formatted
        assert "55,000,000" in formatted

    def test_formats_custom_sizes(self) -> None:
        """Test formatting with custom sizes."""
        stats = create_compression_stats(
            100_000_000,
            50_000_000,
            original_size_mb=440.0,
            compressed_size_mb=220.0,
        )
        formatted = format_compression_stats(stats)
        assert "440.0 MB" in formatted
        assert "220.0 MB" in formatted
