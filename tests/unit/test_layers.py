"""Tests for neural network layers module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.models.layers import (
    VALID_GATING_TYPES,
    VALID_LAYER_TYPES,
    VALID_PROJECTION_TYPES,
    CrossAttentionConfig,
    GatedMLPConfig,
    GatingType,
    LayerConfig,
    LayerStats,
    LayerType,
    MLPConfig,
    ProjectionType,
    calculate_layer_memory,
    calculate_layer_params,
    compare_layer_configs,
    create_cross_attention_config,
    create_gated_mlp_config,
    create_layer_config,
    create_mlp_config,
    estimate_layer_flops,
    format_layer_stats,
    get_gating_type,
    get_layer_type,
    get_projection_type,
    get_recommended_layer_config,
    list_gating_types,
    list_layer_types,
    list_projection_types,
    validate_cross_attention_config,
    validate_gated_mlp_config,
    validate_layer_config,
    validate_layer_stats,
    validate_mlp_config,
)


class TestLayerType:
    """Tests for LayerType enum."""

    def test_mlp_value(self) -> None:
        """Test MLP enum value."""
        assert LayerType.MLP.value == "mlp"

    def test_ffn_value(self) -> None:
        """Test FFN enum value."""
        assert LayerType.FFN.value == "ffn"

    def test_gated_mlp_value(self) -> None:
        """Test GATED_MLP enum value."""
        assert LayerType.GATED_MLP.value == "gated_mlp"

    def test_cross_attention_value(self) -> None:
        """Test CROSS_ATTENTION enum value."""
        assert LayerType.CROSS_ATTENTION.value == "cross_attention"

    def test_self_attention_value(self) -> None:
        """Test SELF_ATTENTION enum value."""
        assert LayerType.SELF_ATTENTION.value == "self_attention"

    def test_conv1d_value(self) -> None:
        """Test CONV1D enum value."""
        assert LayerType.CONV1D.value == "conv1d"

    def test_valid_layer_types_contains_all(self) -> None:
        """Test that VALID_LAYER_TYPES contains all enum values."""
        for lt in LayerType:
            assert lt.value in VALID_LAYER_TYPES


class TestGatingType:
    """Tests for GatingType enum."""

    def test_none_value(self) -> None:
        """Test NONE enum value."""
        assert GatingType.NONE.value == "none"

    def test_swiglu_value(self) -> None:
        """Test SWIGLU enum value."""
        assert GatingType.SWIGLU.value == "swiglu"

    def test_geglu_value(self) -> None:
        """Test GEGLU enum value."""
        assert GatingType.GEGLU.value == "geglu"

    def test_reglu_value(self) -> None:
        """Test REGLU enum value."""
        assert GatingType.REGLU.value == "reglu"

    def test_valid_gating_types_contains_all(self) -> None:
        """Test that VALID_GATING_TYPES contains all enum values."""
        for gt in GatingType:
            assert gt.value in VALID_GATING_TYPES


class TestProjectionType:
    """Tests for ProjectionType enum."""

    def test_linear_value(self) -> None:
        """Test LINEAR enum value."""
        assert ProjectionType.LINEAR.value == "linear"

    def test_low_rank_value(self) -> None:
        """Test LOW_RANK enum value."""
        assert ProjectionType.LOW_RANK.value == "low_rank"

    def test_sparse_value(self) -> None:
        """Test SPARSE enum value."""
        assert ProjectionType.SPARSE.value == "sparse"

    def test_valid_projection_types_contains_all(self) -> None:
        """Test that VALID_PROJECTION_TYPES contains all enum values."""
        for pt in ProjectionType:
            assert pt.value in VALID_PROJECTION_TYPES


class TestMLPConfig:
    """Tests for MLPConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        config = MLPConfig(
            hidden_dim=768,
            intermediate_dim=3072,
            activation="gelu",
            dropout=0.1,
            bias=True,
        )
        assert config.hidden_dim == 768
        assert config.intermediate_dim == 3072
        assert config.activation == "gelu"
        assert config.dropout == pytest.approx(0.1)
        assert config.bias is True

    def test_without_bias(self) -> None:
        """Test creation without bias."""
        config = MLPConfig(
            hidden_dim=4096,
            intermediate_dim=11008,
            activation="silu",
            dropout=0.0,
            bias=False,
        )
        assert config.bias is False

    def test_frozen(self) -> None:
        """Test that MLPConfig is immutable."""
        config = MLPConfig(
            hidden_dim=768,
            intermediate_dim=3072,
            activation="gelu",
            dropout=0.1,
            bias=True,
        )
        with pytest.raises(AttributeError):
            config.hidden_dim = 1024  # type: ignore[misc]


class TestGatedMLPConfig:
    """Tests for GatedMLPConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        mlp = MLPConfig(
            hidden_dim=4096,
            intermediate_dim=11008,
            activation="silu",
            dropout=0.0,
            bias=False,
        )
        config = GatedMLPConfig(
            mlp_config=mlp,
            gating_type=GatingType.SWIGLU,
            gate_dim=11008,
        )
        assert config.mlp_config.hidden_dim == 4096
        assert config.gating_type == GatingType.SWIGLU
        assert config.gate_dim == 11008

    def test_geglu_gating(self) -> None:
        """Test GeGLU gating type."""
        mlp = MLPConfig(
            hidden_dim=2048,
            intermediate_dim=5504,
            activation="gelu",
            dropout=0.0,
            bias=False,
        )
        config = GatedMLPConfig(
            mlp_config=mlp,
            gating_type=GatingType.GEGLU,
            gate_dim=5504,
        )
        assert config.gating_type == GatingType.GEGLU

    def test_frozen(self) -> None:
        """Test that GatedMLPConfig is immutable."""
        mlp = MLPConfig(
            hidden_dim=4096,
            intermediate_dim=11008,
            activation="silu",
            dropout=0.0,
            bias=False,
        )
        config = GatedMLPConfig(
            mlp_config=mlp,
            gating_type=GatingType.SWIGLU,
            gate_dim=11008,
        )
        with pytest.raises(AttributeError):
            config.gate_dim = 8192  # type: ignore[misc]


class TestCrossAttentionConfig:
    """Tests for CrossAttentionConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        config = CrossAttentionConfig(
            hidden_dim=768,
            num_heads=12,
            kv_dim=512,
            dropout=0.1,
        )
        assert config.hidden_dim == 768
        assert config.num_heads == 12
        assert config.kv_dim == 512
        assert config.dropout == pytest.approx(0.1)

    def test_same_kv_dim(self) -> None:
        """Test creation with same kv_dim as hidden_dim."""
        config = CrossAttentionConfig(
            hidden_dim=1024,
            num_heads=16,
            kv_dim=1024,
            dropout=0.0,
        )
        assert config.kv_dim == config.hidden_dim

    def test_frozen(self) -> None:
        """Test that CrossAttentionConfig is immutable."""
        config = CrossAttentionConfig(
            hidden_dim=768,
            num_heads=12,
            kv_dim=512,
            dropout=0.1,
        )
        with pytest.raises(AttributeError):
            config.num_heads = 16  # type: ignore[misc]


class TestLayerConfig:
    """Tests for LayerConfig dataclass."""

    def test_mlp_config(self) -> None:
        """Test MLP layer config."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        assert config.layer_type == LayerType.MLP
        assert config.mlp_config is not None
        assert config.gated_mlp_config is None
        assert config.cross_attention_config is None

    def test_gated_mlp_config(self) -> None:
        """Test Gated MLP layer config."""
        config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        assert config.layer_type == LayerType.GATED_MLP
        assert config.gated_mlp_config is not None
        assert config.mlp_config is None

    def test_cross_attention_config(self) -> None:
        """Test Cross-attention layer config."""
        config = create_layer_config(layer_type="cross_attention", hidden_dim=768)
        assert config.layer_type == LayerType.CROSS_ATTENTION
        assert config.cross_attention_config is not None
        assert config.mlp_config is None

    def test_frozen(self) -> None:
        """Test that LayerConfig is immutable."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        with pytest.raises(AttributeError):
            config.layer_type = LayerType.FFN  # type: ignore[misc]


class TestLayerStats:
    """Tests for LayerStats dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        stats = LayerStats(params=3145728, flops=6291456, memory_mb=12.0)
        assert stats.params == 3145728
        assert stats.flops == 6291456
        assert stats.memory_mb == pytest.approx(12.0)

    def test_frozen(self) -> None:
        """Test that LayerStats is immutable."""
        stats = LayerStats(params=3145728, flops=6291456, memory_mb=12.0)
        with pytest.raises(AttributeError):
            stats.params = 0  # type: ignore[misc]


class TestValidateMLPConfig:
    """Tests for validate_mlp_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = MLPConfig(
            hidden_dim=768,
            intermediate_dim=3072,
            activation="gelu",
            dropout=0.1,
            bias=True,
        )
        validate_mlp_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_mlp_config(None)  # type: ignore[arg-type]

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        config = MLPConfig(
            hidden_dim=0,
            intermediate_dim=3072,
            activation="gelu",
            dropout=0.1,
            bias=True,
        )
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_mlp_config(config)

    def test_negative_hidden_dim_raises_error(self) -> None:
        """Test that negative hidden_dim raises ValueError."""
        config = MLPConfig(
            hidden_dim=-768,
            intermediate_dim=3072,
            activation="gelu",
            dropout=0.1,
            bias=True,
        )
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_mlp_config(config)

    def test_zero_intermediate_dim_raises_error(self) -> None:
        """Test that zero intermediate_dim raises ValueError."""
        config = MLPConfig(
            hidden_dim=768,
            intermediate_dim=0,
            activation="gelu",
            dropout=0.1,
            bias=True,
        )
        with pytest.raises(ValueError, match="intermediate_dim must be positive"):
            validate_mlp_config(config)

    def test_invalid_dropout_raises_error(self) -> None:
        """Test that invalid dropout raises ValueError."""
        config = MLPConfig(
            hidden_dim=768,
            intermediate_dim=3072,
            activation="gelu",
            dropout=1.5,
            bias=True,
        )
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            validate_mlp_config(config)

    def test_negative_dropout_raises_error(self) -> None:
        """Test that negative dropout raises ValueError."""
        config = MLPConfig(
            hidden_dim=768,
            intermediate_dim=3072,
            activation="gelu",
            dropout=-0.1,
            bias=True,
        )
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            validate_mlp_config(config)


class TestValidateGatedMLPConfig:
    """Tests for validate_gated_mlp_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        mlp = MLPConfig(
            hidden_dim=4096,
            intermediate_dim=11008,
            activation="silu",
            dropout=0.0,
            bias=False,
        )
        config = GatedMLPConfig(
            mlp_config=mlp,
            gating_type=GatingType.SWIGLU,
            gate_dim=11008,
        )
        validate_gated_mlp_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_gated_mlp_config(None)  # type: ignore[arg-type]

    def test_zero_gate_dim_raises_error(self) -> None:
        """Test that zero gate_dim raises ValueError."""
        mlp = MLPConfig(
            hidden_dim=4096,
            intermediate_dim=11008,
            activation="silu",
            dropout=0.0,
            bias=False,
        )
        config = GatedMLPConfig(
            mlp_config=mlp,
            gating_type=GatingType.SWIGLU,
            gate_dim=0,
        )
        with pytest.raises(ValueError, match="gate_dim must be positive"):
            validate_gated_mlp_config(config)

    def test_invalid_mlp_config_raises_error(self) -> None:
        """Test that invalid mlp_config raises ValueError."""
        mlp = MLPConfig(
            hidden_dim=0,  # Invalid
            intermediate_dim=11008,
            activation="silu",
            dropout=0.0,
            bias=False,
        )
        config = GatedMLPConfig(
            mlp_config=mlp,
            gating_type=GatingType.SWIGLU,
            gate_dim=11008,
        )
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_gated_mlp_config(config)


class TestValidateCrossAttentionConfig:
    """Tests for validate_cross_attention_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = CrossAttentionConfig(
            hidden_dim=768,
            num_heads=12,
            kv_dim=512,
            dropout=0.1,
        )
        validate_cross_attention_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_cross_attention_config(None)  # type: ignore[arg-type]

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        config = CrossAttentionConfig(
            hidden_dim=0,
            num_heads=12,
            kv_dim=512,
            dropout=0.1,
        )
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_cross_attention_config(config)

    def test_zero_num_heads_raises_error(self) -> None:
        """Test that zero num_heads raises ValueError."""
        config = CrossAttentionConfig(
            hidden_dim=768,
            num_heads=0,
            kv_dim=512,
            dropout=0.1,
        )
        with pytest.raises(ValueError, match="num_heads must be positive"):
            validate_cross_attention_config(config)

    def test_zero_kv_dim_raises_error(self) -> None:
        """Test that zero kv_dim raises ValueError."""
        config = CrossAttentionConfig(
            hidden_dim=768,
            num_heads=12,
            kv_dim=0,
            dropout=0.1,
        )
        with pytest.raises(ValueError, match="kv_dim must be positive"):
            validate_cross_attention_config(config)

    def test_invalid_dropout_raises_error(self) -> None:
        """Test that invalid dropout raises ValueError."""
        config = CrossAttentionConfig(
            hidden_dim=768,
            num_heads=12,
            kv_dim=512,
            dropout=2.0,
        )
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            validate_cross_attention_config(config)


class TestValidateLayerConfig:
    """Tests for validate_layer_config function."""

    def test_valid_mlp_config(self) -> None:
        """Test validation of valid MLP config."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        validate_layer_config(config)  # Should not raise

    def test_valid_gated_mlp_config(self) -> None:
        """Test validation of valid gated MLP config."""
        config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        validate_layer_config(config)  # Should not raise

    def test_valid_cross_attention_config(self) -> None:
        """Test validation of valid cross-attention config."""
        config = create_layer_config(layer_type="cross_attention", hidden_dim=768)
        validate_layer_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_layer_config(None)  # type: ignore[arg-type]

    def test_missing_mlp_config_raises_error(self) -> None:
        """Test that missing mlp_config for MLP type raises error."""
        config = LayerConfig(
            layer_type=LayerType.MLP,
            mlp_config=None,
            gated_mlp_config=None,
            cross_attention_config=None,
        )
        with pytest.raises(ValueError, match="mlp_config required"):
            validate_layer_config(config)

    def test_missing_gated_mlp_config_raises_error(self) -> None:
        """Test that missing gated_mlp_config for GATED_MLP type raises error."""
        config = LayerConfig(
            layer_type=LayerType.GATED_MLP,
            mlp_config=None,
            gated_mlp_config=None,
            cross_attention_config=None,
        )
        with pytest.raises(ValueError, match="gated_mlp_config required"):
            validate_layer_config(config)

    def test_missing_cross_attention_config_raises_error(self) -> None:
        """Test that missing cross_attention_config raises error."""
        config = LayerConfig(
            layer_type=LayerType.CROSS_ATTENTION,
            mlp_config=None,
            gated_mlp_config=None,
            cross_attention_config=None,
        )
        with pytest.raises(ValueError, match="cross_attention_config required"):
            validate_layer_config(config)


class TestValidateLayerStats:
    """Tests for validate_layer_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = LayerStats(params=3145728, flops=6291456, memory_mb=12.0)
        validate_layer_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_layer_stats(None)  # type: ignore[arg-type]

    def test_negative_params_raises_error(self) -> None:
        """Test that negative params raises ValueError."""
        stats = LayerStats(params=-1, flops=6291456, memory_mb=12.0)
        with pytest.raises(ValueError, match="params must be non-negative"):
            validate_layer_stats(stats)

    def test_negative_flops_raises_error(self) -> None:
        """Test that negative flops raises ValueError."""
        stats = LayerStats(params=3145728, flops=-1, memory_mb=12.0)
        with pytest.raises(ValueError, match="flops must be non-negative"):
            validate_layer_stats(stats)

    def test_negative_memory_raises_error(self) -> None:
        """Test that negative memory_mb raises ValueError."""
        stats = LayerStats(params=3145728, flops=6291456, memory_mb=-1.0)
        with pytest.raises(ValueError, match="memory_mb must be non-negative"):
            validate_layer_stats(stats)


class TestCreateMLPConfig:
    """Tests for create_mlp_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_mlp_config()
        assert config.hidden_dim == 768
        assert config.intermediate_dim == 3072  # 4 * 768
        assert config.activation == "gelu"
        assert config.dropout == pytest.approx(0.0)
        assert config.bias is True

    def test_custom_hidden_dim(self) -> None:
        """Test custom hidden_dim."""
        config = create_mlp_config(hidden_dim=1024)
        assert config.hidden_dim == 1024
        assert config.intermediate_dim == 4096  # 4 * 1024

    def test_custom_intermediate_dim(self) -> None:
        """Test custom intermediate_dim."""
        config = create_mlp_config(hidden_dim=768, intermediate_dim=2048)
        assert config.intermediate_dim == 2048

    def test_custom_activation(self) -> None:
        """Test custom activation."""
        config = create_mlp_config(activation="silu")
        assert config.activation == "silu"

    def test_custom_dropout(self) -> None:
        """Test custom dropout."""
        config = create_mlp_config(dropout=0.1)
        assert config.dropout == pytest.approx(0.1)

    def test_no_bias(self) -> None:
        """Test with bias=False."""
        config = create_mlp_config(bias=False)
        assert config.bias is False

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            create_mlp_config(hidden_dim=0)


class TestCreateGatedMLPConfig:
    """Tests for create_gated_mlp_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_gated_mlp_config()
        assert config.mlp_config.hidden_dim == 4096
        assert config.gating_type == GatingType.SWIGLU
        assert config.mlp_config.activation == "silu"
        assert config.mlp_config.bias is False

    def test_custom_hidden_dim(self) -> None:
        """Test custom hidden_dim."""
        config = create_gated_mlp_config(hidden_dim=2048)
        assert config.mlp_config.hidden_dim == 2048

    def test_geglu_gating(self) -> None:
        """Test GeGLU gating type."""
        config = create_gated_mlp_config(gating_type="geglu")
        assert config.gating_type == GatingType.GEGLU

    def test_reglu_gating(self) -> None:
        """Test ReGLU gating type."""
        config = create_gated_mlp_config(gating_type="reglu")
        assert config.gating_type == GatingType.REGLU

    def test_custom_gate_dim(self) -> None:
        """Test custom gate_dim."""
        config = create_gated_mlp_config(gate_dim=8192)
        assert config.gate_dim == 8192

    def test_invalid_gating_type_raises_error(self) -> None:
        """Test that invalid gating_type raises ValueError."""
        with pytest.raises(ValueError, match="gating_type must be one of"):
            create_gated_mlp_config(gating_type="invalid")


class TestCreateCrossAttentionConfig:
    """Tests for create_cross_attention_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_cross_attention_config()
        assert config.hidden_dim == 768
        assert config.num_heads == 12
        assert config.kv_dim == 768
        assert config.dropout == pytest.approx(0.0)

    def test_custom_hidden_dim(self) -> None:
        """Test custom hidden_dim."""
        config = create_cross_attention_config(hidden_dim=1024)
        assert config.hidden_dim == 1024
        assert config.kv_dim == 1024  # Defaults to hidden_dim

    def test_custom_kv_dim(self) -> None:
        """Test custom kv_dim."""
        config = create_cross_attention_config(hidden_dim=768, kv_dim=512)
        assert config.kv_dim == 512

    def test_custom_num_heads(self) -> None:
        """Test custom num_heads."""
        config = create_cross_attention_config(num_heads=16)
        assert config.num_heads == 16

    def test_custom_dropout(self) -> None:
        """Test custom dropout."""
        config = create_cross_attention_config(dropout=0.1)
        assert config.dropout == pytest.approx(0.1)

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            create_cross_attention_config(hidden_dim=0)


class TestCreateLayerConfig:
    """Tests for create_layer_config function."""

    def test_default_mlp(self) -> None:
        """Test default creates MLP config."""
        config = create_layer_config()
        assert config.layer_type == LayerType.MLP
        assert config.mlp_config is not None

    def test_ffn_layer(self) -> None:
        """Test FFN layer type."""
        config = create_layer_config(layer_type="ffn")
        assert config.layer_type == LayerType.FFN
        assert config.mlp_config is not None

    def test_gated_mlp_layer(self) -> None:
        """Test gated MLP layer type."""
        config = create_layer_config(layer_type="gated_mlp")
        assert config.layer_type == LayerType.GATED_MLP
        assert config.gated_mlp_config is not None

    def test_cross_attention_layer(self) -> None:
        """Test cross-attention layer type."""
        config = create_layer_config(layer_type="cross_attention")
        assert config.layer_type == LayerType.CROSS_ATTENTION
        assert config.cross_attention_config is not None

    def test_self_attention_layer(self) -> None:
        """Test self-attention layer type."""
        config = create_layer_config(layer_type="self_attention")
        assert config.layer_type == LayerType.SELF_ATTENTION

    def test_conv1d_layer(self) -> None:
        """Test conv1d layer type."""
        config = create_layer_config(layer_type="conv1d")
        assert config.layer_type == LayerType.CONV1D

    def test_invalid_layer_type_raises_error(self) -> None:
        """Test that invalid layer_type raises ValueError."""
        with pytest.raises(ValueError, match="layer_type must be one of"):
            create_layer_config(layer_type="invalid")

    @given(st.sampled_from(list(VALID_LAYER_TYPES)))
    @settings(max_examples=10)
    def test_all_layer_types_creatable(self, layer_type: str) -> None:
        """Test that all valid layer types can be created."""
        config = create_layer_config(layer_type=layer_type)
        assert config.layer_type.value == layer_type


class TestListLayerTypes:
    """Tests for list_layer_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_layer_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_layer_types()
        assert "mlp" in types
        assert "gated_mlp" in types
        assert "cross_attention" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_layer_types()
        assert types == sorted(types)


class TestListGatingTypes:
    """Tests for list_gating_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_gating_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_gating_types()
        assert "swiglu" in types
        assert "geglu" in types
        assert "reglu" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_gating_types()
        assert types == sorted(types)


class TestListProjectionTypes:
    """Tests for list_projection_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_projection_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_projection_types()
        assert "linear" in types
        assert "low_rank" in types
        assert "sparse" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_projection_types()
        assert types == sorted(types)


class TestGetLayerType:
    """Tests for get_layer_type function."""

    def test_valid_types(self) -> None:
        """Test getting valid layer types."""
        assert get_layer_type("mlp") == LayerType.MLP
        assert get_layer_type("ffn") == LayerType.FFN
        assert get_layer_type("gated_mlp") == LayerType.GATED_MLP

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid layer type"):
            get_layer_type("invalid")


class TestGetGatingType:
    """Tests for get_gating_type function."""

    def test_valid_types(self) -> None:
        """Test getting valid gating types."""
        assert get_gating_type("swiglu") == GatingType.SWIGLU
        assert get_gating_type("geglu") == GatingType.GEGLU
        assert get_gating_type("reglu") == GatingType.REGLU

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid gating type"):
            get_gating_type("invalid")


class TestGetProjectionType:
    """Tests for get_projection_type function."""

    def test_valid_types(self) -> None:
        """Test getting valid projection types."""
        assert get_projection_type("linear") == ProjectionType.LINEAR
        assert get_projection_type("low_rank") == ProjectionType.LOW_RANK
        assert get_projection_type("sparse") == ProjectionType.SPARSE

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid projection type"):
            get_projection_type("invalid")


class TestCalculateLayerParams:
    """Tests for calculate_layer_params function."""

    def test_mlp_params(self) -> None:
        """Test MLP parameter calculation."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        params = calculate_layer_params(config)
        # Up: 768 * 3072 + 3072, Down: 3072 * 768 + 768
        expected = (768 * 3072 + 3072) + (3072 * 768 + 768)
        assert params == expected

    def test_mlp_no_bias_params(self) -> None:
        """Test MLP parameter calculation without bias."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768, bias=False)
        params = calculate_layer_params(config)
        # Up: 768 * 3072, Down: 3072 * 768
        expected = 768 * 3072 + 3072 * 768
        assert params == expected

    def test_gated_mlp_params(self) -> None:
        """Test gated MLP parameter calculation."""
        config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        params = calculate_layer_params(config)
        assert params > 0

    def test_gated_mlp_params_with_bias(self) -> None:
        """Test gated MLP parameter calculation with bias."""
        config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096, bias=True)
        params_bias = calculate_layer_params(config)
        config_no_bias = create_layer_config(
            layer_type="gated_mlp", hidden_dim=4096, bias=False
        )
        params_no_bias = calculate_layer_params(config_no_bias)
        assert params_bias > params_no_bias

    def test_ffn_params(self) -> None:
        """Test FFN layer parameter calculation (alias for MLP)."""
        config = create_layer_config(layer_type="ffn", hidden_dim=768)
        params = calculate_layer_params(config)
        mlp_config = create_layer_config(layer_type="mlp", hidden_dim=768)
        mlp_params = calculate_layer_params(mlp_config)
        assert params == mlp_params

    def test_cross_attention_params(self) -> None:
        """Test cross-attention parameter calculation."""
        config = create_layer_config(layer_type="cross_attention", hidden_dim=768)
        params = calculate_layer_params(config)
        # Q: 768*768, K: 768*768, V: 768*768, O: 768*768
        expected = 4 * 768 * 768
        assert params == expected

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_layer_params(None)  # type: ignore[arg-type]

    def test_self_attention_returns_zero(self) -> None:
        """Test self-attention returns 0 (no config support)."""
        config = create_layer_config(layer_type="self_attention")
        params = calculate_layer_params(config)
        assert params == 0

    def test_conv1d_returns_zero(self) -> None:
        """Test conv1d returns 0 (no config support)."""
        config = create_layer_config(layer_type="conv1d")
        params = calculate_layer_params(config)
        assert params == 0


class TestEstimateLayerFlops:
    """Tests for estimate_layer_flops function."""

    def test_mlp_flops(self) -> None:
        """Test MLP FLOPs estimation."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        flops = estimate_layer_flops(config)
        assert flops > 0

    def test_gated_mlp_flops(self) -> None:
        """Test gated MLP FLOPs estimation."""
        config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        flops = estimate_layer_flops(config)
        assert flops > 0

    def test_cross_attention_flops(self) -> None:
        """Test cross-attention FLOPs estimation."""
        config = create_layer_config(layer_type="cross_attention", hidden_dim=768)
        flops = estimate_layer_flops(config)
        assert flops > 0

    def test_flops_scale_with_batch_size(self) -> None:
        """Test that FLOPs scale with batch size."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        flops_1 = estimate_layer_flops(config, batch_size=1)
        flops_2 = estimate_layer_flops(config, batch_size=2)
        assert flops_2 == 2 * flops_1

    def test_flops_scale_with_seq_length(self) -> None:
        """Test that FLOPs scale with sequence length."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        flops_512 = estimate_layer_flops(config, seq_length=512)
        flops_1024 = estimate_layer_flops(config, seq_length=1024)
        assert flops_1024 == 2 * flops_512

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            estimate_layer_flops(None)  # type: ignore[arg-type]

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = create_layer_config(layer_type="mlp")
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_layer_flops(config, batch_size=0)

    def test_ffn_flops(self) -> None:
        """Test FFN FLOPs estimation (same as MLP)."""
        config = create_layer_config(layer_type="ffn", hidden_dim=768)
        flops = estimate_layer_flops(config)
        assert flops > 0

    def test_self_attention_flops_returns_zero(self) -> None:
        """Test self-attention returns 0 FLOPs (no config support)."""
        config = create_layer_config(layer_type="self_attention")
        flops = estimate_layer_flops(config)
        assert flops == 0

    def test_conv1d_flops_returns_zero(self) -> None:
        """Test conv1d returns 0 FLOPs (no config support)."""
        config = create_layer_config(layer_type="conv1d")
        flops = estimate_layer_flops(config)
        assert flops == 0

    def test_zero_seq_length_raises_error(self) -> None:
        """Test that zero seq_length raises ValueError."""
        config = create_layer_config(layer_type="mlp")
        with pytest.raises(ValueError, match="seq_length must be positive"):
            estimate_layer_flops(config, seq_length=0)


class TestCalculateLayerMemory:
    """Tests for calculate_layer_memory function."""

    def test_mlp_memory(self) -> None:
        """Test MLP memory calculation."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        memory = calculate_layer_memory(config)
        assert memory > 0

    def test_gated_mlp_memory(self) -> None:
        """Test gated MLP memory calculation."""
        config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        memory = calculate_layer_memory(config)
        assert memory > 0

    def test_cross_attention_memory(self) -> None:
        """Test cross-attention memory calculation."""
        config = create_layer_config(layer_type="cross_attention", hidden_dim=768)
        memory = calculate_layer_memory(config)
        assert memory > 0

    def test_memory_scale_with_batch_size(self) -> None:
        """Test that memory scales with batch size."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        mem_1 = calculate_layer_memory(config, batch_size=1)
        mem_2 = calculate_layer_memory(config, batch_size=2)
        assert mem_2 > mem_1

    def test_fp32_more_memory_than_fp16(self) -> None:
        """Test that fp32 uses more memory than fp16."""
        config = create_layer_config(layer_type="mlp", hidden_dim=768)
        mem_fp16 = calculate_layer_memory(config, dtype_bytes=2)
        mem_fp32 = calculate_layer_memory(config, dtype_bytes=4)
        assert mem_fp32 > mem_fp16

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_layer_memory(None)  # type: ignore[arg-type]

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = create_layer_config(layer_type="mlp")
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_layer_memory(config, batch_size=0)

    def test_zero_seq_length_raises_error(self) -> None:
        """Test that zero seq_length raises ValueError."""
        config = create_layer_config(layer_type="mlp")
        with pytest.raises(ValueError, match="seq_length must be positive"):
            calculate_layer_memory(config, seq_length=0)

    def test_invalid_dtype_bytes_raises_error(self) -> None:
        """Test that invalid dtype_bytes raises ValueError."""
        config = create_layer_config(layer_type="mlp")
        with pytest.raises(ValueError, match="dtype_bytes must be"):
            calculate_layer_memory(config, dtype_bytes=3)

    def test_ffn_memory(self) -> None:
        """Test FFN memory calculation (same as MLP)."""
        config = create_layer_config(layer_type="ffn", hidden_dim=768)
        memory = calculate_layer_memory(config)
        assert memory > 0

    def test_self_attention_memory(self) -> None:
        """Test self-attention memory calculation (params only)."""
        config = create_layer_config(layer_type="self_attention")
        memory = calculate_layer_memory(config)
        # Should return memory for params (0) only
        assert memory >= 0

    def test_conv1d_memory(self) -> None:
        """Test conv1d memory calculation (params only)."""
        config = create_layer_config(layer_type="conv1d")
        memory = calculate_layer_memory(config)
        # Should return memory for params (0) only
        assert memory >= 0


class TestCompareLayerConfigs:
    """Tests for compare_layer_configs function."""

    def test_basic_comparison(self) -> None:
        """Test basic comparison."""
        mlp = create_layer_config(layer_type="mlp", hidden_dim=768)
        gated = create_layer_config(layer_type="gated_mlp", hidden_dim=768)
        results = compare_layer_configs([mlp, gated])
        assert "mlp" in results
        assert "gated_mlp" in results

    def test_gated_mlp_more_params(self) -> None:
        """Test that gated MLP has more params than regular MLP."""
        mlp = create_layer_config(layer_type="mlp", hidden_dim=768)
        gated = create_layer_config(layer_type="gated_mlp", hidden_dim=768)
        results = compare_layer_configs([mlp, gated])
        assert results["gated_mlp"].params > results["mlp"].params

    def test_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compare_layer_configs([])

    def test_none_config_raises_error(self) -> None:
        """Test that None config in list raises ValueError."""
        mlp = create_layer_config(layer_type="mlp")
        with pytest.raises(ValueError, match="cannot be None"):
            compare_layer_configs([mlp, None])  # type: ignore[list-item]


class TestFormatLayerStats:
    """Tests for format_layer_stats function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        stats = LayerStats(params=3145728, flops=6291456, memory_mb=12.0)
        formatted = format_layer_stats(stats)
        assert "Parameters:" in formatted
        assert "FLOPs:" in formatted
        assert "Memory:" in formatted

    def test_large_numbers(self) -> None:
        """Test formatting of large numbers."""
        stats = LayerStats(params=1000000000, flops=2000000000000, memory_mb=1024.0)
        formatted = format_layer_stats(stats)
        assert "B" in formatted or "G" in formatted  # Billion
        assert "T" in formatted  # Trillion FLOPs

    def test_small_numbers(self) -> None:
        """Test formatting of small numbers."""
        stats = LayerStats(params=1000, flops=2000, memory_mb=0.5)
        formatted = format_layer_stats(stats)
        assert "K" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_layer_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedLayerConfig:
    """Tests for get_recommended_layer_config function."""

    def test_llm_default(self) -> None:
        """Test LLM default recommendation."""
        config = get_recommended_layer_config("llm")
        assert config.layer_type == LayerType.GATED_MLP

    def test_llm_efficiency_priority(self) -> None:
        """Test LLM with efficiency priority."""
        config = get_recommended_layer_config("llm", efficiency_priority=True)
        assert config.layer_type == LayerType.MLP

    def test_encoder_default(self) -> None:
        """Test encoder default recommendation."""
        config = get_recommended_layer_config("encoder")
        assert config.layer_type == LayerType.MLP

    def test_decoder_default(self) -> None:
        """Test decoder default recommendation."""
        config = get_recommended_layer_config("decoder")
        assert config.layer_type == LayerType.GATED_MLP

    def test_decoder_efficiency_priority(self) -> None:
        """Test decoder with efficiency priority."""
        config = get_recommended_layer_config("decoder", efficiency_priority=True)
        assert config.layer_type == LayerType.MLP

    def test_custom_hidden_dim(self) -> None:
        """Test custom hidden_dim."""
        config = get_recommended_layer_config("llm", hidden_dim=8192)
        assert config.gated_mlp_config is not None
        assert config.gated_mlp_config.mlp_config.hidden_dim == 8192

    def test_invalid_model_type_raises_error(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_recommended_layer_config("invalid")


class TestPropertyBasedTests:
    """Property-based tests using hypothesis."""

    @given(
        st.integers(min_value=64, max_value=8192),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=4096),
    )
    @settings(max_examples=20)
    def test_mlp_memory_always_positive(
        self, hidden_dim: int, batch_size: int, seq_length: int
    ) -> None:
        """Test that MLP memory is always positive."""
        config = create_layer_config(layer_type="mlp", hidden_dim=hidden_dim)
        memory = calculate_layer_memory(
            config, batch_size=batch_size, seq_length=seq_length
        )
        assert memory > 0

    @given(
        st.integers(min_value=64, max_value=8192),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=4096),
    )
    @settings(max_examples=20)
    def test_mlp_flops_always_positive(
        self, hidden_dim: int, batch_size: int, seq_length: int
    ) -> None:
        """Test that MLP FLOPs are always positive."""
        config = create_layer_config(layer_type="mlp", hidden_dim=hidden_dim)
        flops = estimate_layer_flops(
            config, batch_size=batch_size, seq_length=seq_length
        )
        assert flops > 0

    @given(st.integers(min_value=64, max_value=8192))
    @settings(max_examples=20)
    def test_mlp_params_always_positive(self, hidden_dim: int) -> None:
        """Test that MLP params are always positive."""
        config = create_layer_config(layer_type="mlp", hidden_dim=hidden_dim)
        params = calculate_layer_params(config)
        assert params > 0

    @given(st.sampled_from(["mlp", "ffn", "gated_mlp", "cross_attention"]))
    @settings(max_examples=10)
    def test_all_supported_layer_types_have_params(self, layer_type: str) -> None:
        """Test that all supported layer types have positive params."""
        config = create_layer_config(layer_type=layer_type)
        params = calculate_layer_params(config)
        assert params > 0
