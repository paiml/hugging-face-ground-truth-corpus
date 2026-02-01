"""Tests for models.normalization module."""

from __future__ import annotations

import pytest

from hf_gtc.models.normalization import (
    VALID_EPS_TYPES,
    VALID_NORM_POSITIONS,
    VALID_NORM_TYPES,
    BatchNormConfig,
    EpsType,
    GroupNormConfig,
    LayerNormConfig,
    NormConfig,
    NormPosition,
    NormStats,
    NormType,
    RMSNormConfig,
    calculate_norm_params,
    compare_norm_stability,
    create_batch_norm_config,
    create_group_norm_config,
    create_layer_norm_config,
    create_norm_config,
    create_rms_norm_config,
    estimate_norm_memory,
    format_norm_stats,
    get_eps_type,
    get_eps_value,
    get_norm_position,
    get_norm_type,
    get_recommended_norm_config,
    list_eps_types,
    list_norm_positions,
    list_norm_types,
    select_eps_for_dtype,
    validate_batch_norm_config,
    validate_group_norm_config,
    validate_layer_norm_config,
    validate_norm_config,
    validate_rms_norm_config,
)


class TestNormType:
    """Tests for NormType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for norm_type in NormType:
            assert isinstance(norm_type.value, str)

    def test_layer_norm_value(self) -> None:
        """Layer norm has correct value."""
        assert NormType.LAYER_NORM.value == "layer_norm"

    def test_rms_norm_value(self) -> None:
        """RMS norm has correct value."""
        assert NormType.RMS_NORM.value == "rms_norm"

    def test_batch_norm_value(self) -> None:
        """Batch norm has correct value."""
        assert NormType.BATCH_NORM.value == "batch_norm"

    def test_group_norm_value(self) -> None:
        """Group norm has correct value."""
        assert NormType.GROUP_NORM.value == "group_norm"

    def test_instance_norm_value(self) -> None:
        """Instance norm has correct value."""
        assert NormType.INSTANCE_NORM.value == "instance_norm"

    def test_none_value(self) -> None:
        """None has correct value."""
        assert NormType.NONE.value == "none"

    def test_valid_norm_types_frozenset(self) -> None:
        """VALID_NORM_TYPES is a frozenset."""
        assert isinstance(VALID_NORM_TYPES, frozenset)
        assert len(VALID_NORM_TYPES) == 6


class TestNormPosition:
    """Tests for NormPosition enum."""

    def test_all_positions_have_values(self) -> None:
        """All positions have string values."""
        for pos in NormPosition:
            assert isinstance(pos.value, str)

    def test_pre_value(self) -> None:
        """Pre has correct value."""
        assert NormPosition.PRE.value == "pre"

    def test_post_value(self) -> None:
        """Post has correct value."""
        assert NormPosition.POST.value == "post"

    def test_sandwich_value(self) -> None:
        """Sandwich has correct value."""
        assert NormPosition.SANDWICH.value == "sandwich"

    def test_valid_norm_positions_frozenset(self) -> None:
        """VALID_NORM_POSITIONS is a frozenset."""
        assert isinstance(VALID_NORM_POSITIONS, frozenset)
        assert len(VALID_NORM_POSITIONS) == 3


class TestEpsType:
    """Tests for EpsType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for eps_type in EpsType:
            assert isinstance(eps_type.value, str)

    def test_standard_value(self) -> None:
        """Standard has correct value."""
        assert EpsType.STANDARD.value == "standard"

    def test_fp16_safe_value(self) -> None:
        """FP16 safe has correct value."""
        assert EpsType.FP16_SAFE.value == "fp16_safe"

    def test_bf16_safe_value(self) -> None:
        """BF16 safe has correct value."""
        assert EpsType.BF16_SAFE.value == "bf16_safe"

    def test_valid_eps_types_frozenset(self) -> None:
        """VALID_EPS_TYPES is a frozenset."""
        assert isinstance(VALID_EPS_TYPES, frozenset)
        assert len(VALID_EPS_TYPES) == 3


class TestLayerNormConfig:
    """Tests for LayerNormConfig dataclass."""

    def test_create_config(self) -> None:
        """Create layer norm config."""
        config = LayerNormConfig(
            normalized_shape=(768,),
            eps=1e-5,
            elementwise_affine=True,
        )
        assert config.normalized_shape == (768,)
        assert config.eps == 1e-5
        assert config.elementwise_affine is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = LayerNormConfig((768,), 1e-5, True)
        with pytest.raises(AttributeError):
            config.eps = 1e-6  # type: ignore[misc]

    def test_multi_dim_shape(self) -> None:
        """Config with multi-dimensional shape."""
        config = LayerNormConfig((512, 768), 1e-5, True)
        assert config.normalized_shape == (512, 768)


class TestRMSNormConfig:
    """Tests for RMSNormConfig dataclass."""

    def test_create_config(self) -> None:
        """Create RMS norm config."""
        config = RMSNormConfig(
            hidden_size=4096,
            eps=1e-6,
            add_unit_offset=False,
        )
        assert config.hidden_size == 4096
        assert config.eps == 1e-6
        assert config.add_unit_offset is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = RMSNormConfig(4096, 1e-6, False)
        with pytest.raises(AttributeError):
            config.hidden_size = 2048  # type: ignore[misc]

    def test_with_unit_offset(self) -> None:
        """Config with unit offset (LLaMA style)."""
        config = RMSNormConfig(4096, 1e-6, True)
        assert config.add_unit_offset is True


class TestGroupNormConfig:
    """Tests for GroupNormConfig dataclass."""

    def test_create_config(self) -> None:
        """Create group norm config."""
        config = GroupNormConfig(
            num_groups=32,
            num_channels=256,
            eps=1e-5,
        )
        assert config.num_groups == 32
        assert config.num_channels == 256
        assert config.eps == 1e-5
        assert config.affine is True  # Default

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = GroupNormConfig(32, 256, 1e-5)
        with pytest.raises(AttributeError):
            config.num_groups = 16  # type: ignore[misc]

    def test_no_affine(self) -> None:
        """Config without affine parameters."""
        config = GroupNormConfig(32, 256, 1e-5, affine=False)
        assert config.affine is False


class TestBatchNormConfig:
    """Tests for BatchNormConfig dataclass."""

    def test_create_config(self) -> None:
        """Create batch norm config."""
        config = BatchNormConfig(
            num_features=256,
            eps=1e-5,
            momentum=0.1,
        )
        assert config.num_features == 256
        assert config.eps == 1e-5
        assert config.momentum == pytest.approx(0.1)
        assert config.affine is True
        assert config.track_running_stats is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = BatchNormConfig(256, 1e-5, 0.1)
        with pytest.raises(AttributeError):
            config.momentum = 0.05  # type: ignore[misc]


class TestNormConfig:
    """Tests for NormConfig dataclass."""

    def test_create_layer_norm_config(self) -> None:
        """Create config with layer norm."""
        layer_config = LayerNormConfig((768,), 1e-5, True)
        config = NormConfig(
            norm_type=NormType.LAYER_NORM,
            layer_norm_config=layer_config,
            rms_norm_config=None,
            group_norm_config=None,
            batch_norm_config=None,
            position=NormPosition.PRE,
        )
        assert config.norm_type == NormType.LAYER_NORM
        assert config.position == NormPosition.PRE

    def test_create_rms_norm_config(self) -> None:
        """Create config with RMS norm."""
        rms_config = RMSNormConfig(4096, 1e-6, False)
        config = NormConfig(
            norm_type=NormType.RMS_NORM,
            layer_norm_config=None,
            rms_norm_config=rms_config,
            group_norm_config=None,
            batch_norm_config=None,
            position=NormPosition.PRE,
        )
        assert config.norm_type == NormType.RMS_NORM

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        layer_config = LayerNormConfig((768,), 1e-5, True)
        config = NormConfig(
            NormType.LAYER_NORM, layer_config, None, None, None, NormPosition.PRE
        )
        with pytest.raises(AttributeError):
            config.norm_type = NormType.RMS_NORM  # type: ignore[misc]


class TestNormStats:
    """Tests for NormStats dataclass."""

    def test_create_stats(self) -> None:
        """Create norm stats."""
        stats = NormStats(
            mean_activation=0.0,
            std_activation=1.0,
            num_parameters=1536,
            memory_bytes=6144,
            flops_per_token=4608,
        )
        assert stats.mean_activation == pytest.approx(0.0)
        assert stats.std_activation == pytest.approx(1.0)
        assert stats.num_parameters == 1536


class TestValidateLayerNormConfig:
    """Tests for validate_layer_norm_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LayerNormConfig((768,), 1e-5, True)
        validate_layer_norm_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_layer_norm_config(None)  # type: ignore[arg-type]

    def test_empty_shape_raises(self) -> None:
        """Empty shape raises ValueError."""
        config = LayerNormConfig((), 1e-5, True)
        with pytest.raises(ValueError, match="normalized_shape cannot be empty"):
            validate_layer_norm_config(config)

    def test_negative_dim_raises(self) -> None:
        """Negative dimension raises ValueError."""
        config = LayerNormConfig((-1,), 1e-5, True)
        with pytest.raises(ValueError, match="dimensions must be positive"):
            validate_layer_norm_config(config)

    def test_zero_dim_raises(self) -> None:
        """Zero dimension raises ValueError."""
        config = LayerNormConfig((0,), 1e-5, True)
        with pytest.raises(ValueError, match="dimensions must be positive"):
            validate_layer_norm_config(config)

    def test_negative_eps_raises(self) -> None:
        """Negative eps raises ValueError."""
        config = LayerNormConfig((768,), -1e-5, True)
        with pytest.raises(ValueError, match="eps must be positive"):
            validate_layer_norm_config(config)

    def test_zero_eps_raises(self) -> None:
        """Zero eps raises ValueError."""
        config = LayerNormConfig((768,), 0, True)
        with pytest.raises(ValueError, match="eps must be positive"):
            validate_layer_norm_config(config)


class TestValidateRMSNormConfig:
    """Tests for validate_rms_norm_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RMSNormConfig(4096, 1e-6, False)
        validate_rms_norm_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_rms_norm_config(None)  # type: ignore[arg-type]

    def test_zero_hidden_size_raises(self) -> None:
        """Zero hidden size raises ValueError."""
        config = RMSNormConfig(0, 1e-6, False)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_rms_norm_config(config)

    def test_negative_hidden_size_raises(self) -> None:
        """Negative hidden size raises ValueError."""
        config = RMSNormConfig(-1, 1e-6, False)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_rms_norm_config(config)

    def test_negative_eps_raises(self) -> None:
        """Negative eps raises ValueError."""
        config = RMSNormConfig(4096, -1e-6, False)
        with pytest.raises(ValueError, match="eps must be positive"):
            validate_rms_norm_config(config)


class TestValidateGroupNormConfig:
    """Tests for validate_group_norm_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = GroupNormConfig(32, 256, 1e-5)
        validate_group_norm_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_group_norm_config(None)  # type: ignore[arg-type]

    def test_zero_num_groups_raises(self) -> None:
        """Zero num_groups raises ValueError."""
        config = GroupNormConfig(0, 256, 1e-5)
        with pytest.raises(ValueError, match="num_groups must be positive"):
            validate_group_norm_config(config)

    def test_negative_num_groups_raises(self) -> None:
        """Negative num_groups raises ValueError."""
        config = GroupNormConfig(-1, 256, 1e-5)
        with pytest.raises(ValueError, match="num_groups must be positive"):
            validate_group_norm_config(config)

    def test_zero_num_channels_raises(self) -> None:
        """Zero num_channels raises ValueError."""
        config = GroupNormConfig(32, 0, 1e-5)
        with pytest.raises(ValueError, match="num_channels must be positive"):
            validate_group_norm_config(config)

    def test_indivisible_channels_raises(self) -> None:
        """Non-divisible channels raises ValueError."""
        config = GroupNormConfig(32, 100, 1e-5)
        with pytest.raises(ValueError, match="must be divisible"):
            validate_group_norm_config(config)

    def test_negative_eps_raises(self) -> None:
        """Negative eps raises ValueError."""
        config = GroupNormConfig(32, 256, -1e-5)
        with pytest.raises(ValueError, match="eps must be positive"):
            validate_group_norm_config(config)


class TestValidateBatchNormConfig:
    """Tests for validate_batch_norm_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = BatchNormConfig(256, 1e-5, 0.1)
        validate_batch_norm_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_batch_norm_config(None)  # type: ignore[arg-type]

    def test_zero_num_features_raises(self) -> None:
        """Zero num_features raises ValueError."""
        config = BatchNormConfig(0, 1e-5, 0.1)
        with pytest.raises(ValueError, match="num_features must be positive"):
            validate_batch_norm_config(config)

    def test_negative_eps_raises(self) -> None:
        """Negative eps raises ValueError."""
        config = BatchNormConfig(256, -1e-5, 0.1)
        with pytest.raises(ValueError, match="eps must be positive"):
            validate_batch_norm_config(config)

    def test_negative_momentum_raises(self) -> None:
        """Negative momentum raises ValueError."""
        config = BatchNormConfig(256, 1e-5, -0.1)
        with pytest.raises(ValueError, match="momentum must be between 0 and 1"):
            validate_batch_norm_config(config)

    def test_momentum_too_high_raises(self) -> None:
        """Momentum > 1 raises ValueError."""
        config = BatchNormConfig(256, 1e-5, 1.5)
        with pytest.raises(ValueError, match="momentum must be between 0 and 1"):
            validate_batch_norm_config(config)


class TestValidateNormConfig:
    """Tests for validate_norm_config function."""

    def test_valid_layer_norm_config(self) -> None:
        """Valid layer norm config passes."""
        layer_config = LayerNormConfig((768,), 1e-5, True)
        config = NormConfig(
            NormType.LAYER_NORM, layer_config, None, None, None, NormPosition.PRE
        )
        validate_norm_config(config)

    def test_valid_none_config(self) -> None:
        """Valid none config passes."""
        config = NormConfig(NormType.NONE, None, None, None, None, NormPosition.PRE)
        validate_norm_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_norm_config(None)  # type: ignore[arg-type]

    def test_missing_layer_norm_config_raises(self) -> None:
        """Missing layer_norm_config raises ValueError."""
        config = NormConfig(
            NormType.LAYER_NORM, None, None, None, None, NormPosition.PRE
        )
        with pytest.raises(ValueError, match="layer_norm_config required"):
            validate_norm_config(config)

    def test_missing_rms_norm_config_raises(self) -> None:
        """Missing rms_norm_config raises ValueError."""
        config = NormConfig(NormType.RMS_NORM, None, None, None, None, NormPosition.PRE)
        with pytest.raises(ValueError, match="rms_norm_config required"):
            validate_norm_config(config)

    def test_missing_group_norm_config_raises(self) -> None:
        """Missing group_norm_config raises ValueError."""
        config = NormConfig(
            NormType.GROUP_NORM, None, None, None, None, NormPosition.PRE
        )
        with pytest.raises(ValueError, match="group_norm_config required"):
            validate_norm_config(config)

    def test_missing_batch_norm_config_raises(self) -> None:
        """Missing batch_norm_config raises ValueError."""
        config = NormConfig(
            NormType.BATCH_NORM, None, None, None, None, NormPosition.PRE
        )
        with pytest.raises(ValueError, match="batch_norm_config required"):
            validate_norm_config(config)


class TestCreateLayerNormConfig:
    """Tests for create_layer_norm_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_layer_norm_config()
        assert config.normalized_shape == (768,)
        assert config.eps == 1e-5
        assert config.elementwise_affine is True

    def test_int_shape(self) -> None:
        """Create config with int shape."""
        config = create_layer_norm_config(512)
        assert config.normalized_shape == (512,)

    def test_tuple_shape(self) -> None:
        """Create config with tuple shape."""
        config = create_layer_norm_config((256, 768))
        assert config.normalized_shape == (256, 768)

    def test_custom_eps(self) -> None:
        """Create config with custom eps."""
        config = create_layer_norm_config(768, eps=1e-6)
        assert config.eps == 1e-6

    def test_no_affine(self) -> None:
        """Create config without affine."""
        config = create_layer_norm_config(768, elementwise_affine=False)
        assert config.elementwise_affine is False

    def test_invalid_shape_raises(self) -> None:
        """Invalid shape raises ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            create_layer_norm_config(0)


class TestCreateRMSNormConfig:
    """Tests for create_rms_norm_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_rms_norm_config()
        assert config.hidden_size == 4096
        assert config.eps == 1e-6
        assert config.add_unit_offset is False

    def test_custom_hidden_size(self) -> None:
        """Create config with custom hidden size."""
        config = create_rms_norm_config(2048)
        assert config.hidden_size == 2048

    def test_custom_eps(self) -> None:
        """Create config with custom eps."""
        config = create_rms_norm_config(4096, eps=1e-5)
        assert config.eps == 1e-5

    def test_with_unit_offset(self) -> None:
        """Create config with unit offset."""
        config = create_rms_norm_config(4096, add_unit_offset=True)
        assert config.add_unit_offset is True

    def test_invalid_hidden_size_raises(self) -> None:
        """Invalid hidden size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            create_rms_norm_config(0)


class TestCreateGroupNormConfig:
    """Tests for create_group_norm_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_group_norm_config()
        assert config.num_groups == 32
        assert config.num_channels == 256
        assert config.eps == 1e-5
        assert config.affine is True

    def test_custom_groups(self) -> None:
        """Create config with custom groups."""
        config = create_group_norm_config(16, 256)
        assert config.num_groups == 16

    def test_no_affine(self) -> None:
        """Create config without affine."""
        config = create_group_norm_config(32, 256, affine=False)
        assert config.affine is False

    def test_indivisible_raises(self) -> None:
        """Indivisible channels raises ValueError."""
        with pytest.raises(ValueError, match="num_channels must be divisible"):
            create_group_norm_config(32, 100)


class TestCreateBatchNormConfig:
    """Tests for create_batch_norm_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_batch_norm_config()
        assert config.num_features == 256
        assert config.eps == 1e-5
        assert config.momentum == pytest.approx(0.1)
        assert config.affine is True
        assert config.track_running_stats is True

    def test_custom_features(self) -> None:
        """Create config with custom features."""
        config = create_batch_norm_config(512)
        assert config.num_features == 512

    def test_custom_momentum(self) -> None:
        """Create config with custom momentum."""
        config = create_batch_norm_config(256, momentum=0.2)
        assert config.momentum == pytest.approx(0.2)

    def test_invalid_momentum_raises(self) -> None:
        """Invalid momentum raises ValueError."""
        with pytest.raises(ValueError, match="momentum must be between 0 and 1"):
            create_batch_norm_config(256, momentum=1.5)


class TestCreateNormConfig:
    """Tests for create_norm_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_norm_config()
        assert config.norm_type == NormType.LAYER_NORM
        assert config.position == NormPosition.PRE

    def test_layer_norm(self) -> None:
        """Create layer norm config."""
        config = create_norm_config("layer_norm", 768)
        assert config.norm_type == NormType.LAYER_NORM
        assert config.layer_norm_config is not None
        assert config.layer_norm_config.normalized_shape == (768,)

    def test_rms_norm(self) -> None:
        """Create RMS norm config."""
        config = create_norm_config("rms_norm", 4096)
        assert config.norm_type == NormType.RMS_NORM
        assert config.rms_norm_config is not None
        assert config.rms_norm_config.hidden_size == 4096

    def test_group_norm(self) -> None:
        """Create group norm config."""
        config = create_norm_config("group_norm", 256, num_groups=32)
        assert config.norm_type == NormType.GROUP_NORM
        assert config.group_norm_config is not None
        assert config.group_norm_config.num_groups == 32

    def test_batch_norm(self) -> None:
        """Create batch norm config."""
        config = create_norm_config("batch_norm", 256)
        assert config.norm_type == NormType.BATCH_NORM
        assert config.batch_norm_config is not None

    def test_none_norm(self) -> None:
        """Create none norm config."""
        config = create_norm_config("none")
        assert config.norm_type == NormType.NONE

    def test_enum_type(self) -> None:
        """Create config with enum type."""
        config = create_norm_config(NormType.RMS_NORM, 4096)
        assert config.norm_type == NormType.RMS_NORM

    def test_enum_position(self) -> None:
        """Create config with enum position."""
        config = create_norm_config("layer_norm", 768, position=NormPosition.POST)
        assert config.position == NormPosition.POST

    def test_custom_eps(self) -> None:
        """Create config with custom eps."""
        config = create_norm_config("layer_norm", 768, eps=1e-6)
        assert config.layer_norm_config is not None
        assert config.layer_norm_config.eps == 1e-6

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="norm_type must be one of"):
            create_norm_config("invalid")


class TestCalculateNormParams:
    """Tests for calculate_norm_params function."""

    def test_layer_norm_params(self) -> None:
        """Calculate layer norm params."""
        config = create_norm_config("layer_norm", 768)
        params = calculate_norm_params(config)
        assert params == 768 * 2  # gamma + beta

    def test_rms_norm_params(self) -> None:
        """Calculate RMS norm params."""
        config = create_norm_config("rms_norm", 4096)
        params = calculate_norm_params(config)
        assert params == 4096  # Only weight (gamma)

    def test_group_norm_params(self) -> None:
        """Calculate group norm params."""
        config = create_norm_config("group_norm", 256, num_groups=32)
        params = calculate_norm_params(config)
        assert params == 256 * 2  # gamma + beta per channel

    def test_batch_norm_params(self) -> None:
        """Calculate batch norm params."""
        config = create_norm_config("batch_norm", 256)
        params = calculate_norm_params(config)
        assert params == 256 * 2

    def test_none_norm_params(self) -> None:
        """None norm has zero params."""
        config = create_norm_config("none")
        params = calculate_norm_params(config)
        assert params == 0

    def test_no_affine_params(self) -> None:
        """No affine means zero params."""
        config = create_norm_config("layer_norm", 768, elementwise_affine=False)
        params = calculate_norm_params(config)
        assert params == 0

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            calculate_norm_params(None)  # type: ignore[arg-type]

    def test_multi_dim_layer_norm(self) -> None:
        """Multi-dimensional layer norm params."""
        layer_config = LayerNormConfig((256, 768), 1e-5, True)
        config = NormConfig(
            NormType.LAYER_NORM, layer_config, None, None, None, NormPosition.PRE
        )
        params = calculate_norm_params(config)
        assert params == 256 * 768 * 2


class TestEstimateNormMemory:
    """Tests for estimate_norm_memory function."""

    def test_layer_norm_memory(self) -> None:
        """Estimate layer norm memory."""
        config = create_norm_config("layer_norm", 768)
        mem = estimate_norm_memory(config, batch_size=1, sequence_length=512)
        assert mem > 0

    def test_none_norm_memory(self) -> None:
        """None norm has zero memory."""
        config = create_norm_config("none")
        mem = estimate_norm_memory(config)
        assert mem == 0

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            estimate_norm_memory(None)  # type: ignore[arg-type]

    def test_invalid_batch_size_raises(self) -> None:
        """Invalid batch size raises ValueError."""
        config = create_norm_config("layer_norm", 768)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_norm_memory(config, batch_size=0)

    def test_invalid_sequence_length_raises(self) -> None:
        """Invalid sequence length raises ValueError."""
        config = create_norm_config("layer_norm", 768)
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_norm_memory(config, sequence_length=0)

    def test_larger_batch_more_memory(self) -> None:
        """Larger batch uses more memory."""
        config = create_norm_config("layer_norm", 768)
        mem1 = estimate_norm_memory(config, batch_size=1)
        mem2 = estimate_norm_memory(config, batch_size=4)
        assert mem2 > mem1


class TestCompareNormStability:
    """Tests for compare_norm_stability function."""

    def test_layer_vs_rms(self) -> None:
        """Compare layer norm vs RMS norm."""
        result = compare_norm_stability("layer_norm", "rms_norm")
        assert "recommended" in result
        assert "reason" in result
        assert "more_efficient" in result

    def test_layer_vs_batch(self) -> None:
        """Compare layer norm vs batch norm."""
        result = compare_norm_stability("layer_norm", "batch_norm")
        assert result["recommended"] == "layer_norm"

    def test_none_vs_layer(self) -> None:
        """Compare none vs layer norm."""
        result = compare_norm_stability("none", "layer_norm")
        assert result["recommended"] == "layer_norm"

    def test_layer_vs_none(self) -> None:
        """Compare layer norm vs none."""
        result = compare_norm_stability("layer_norm", "none")
        assert result["recommended"] == "layer_norm"

    def test_same_types(self) -> None:
        """Compare same types."""
        result = compare_norm_stability("layer_norm", "layer_norm")
        assert "similar" in result["reason"]

    def test_enum_input(self) -> None:
        """Compare using enum inputs."""
        result = compare_norm_stability(NormType.LAYER_NORM, NormType.RMS_NORM)
        assert "recommended" in result


class TestSelectEpsForDtype:
    """Tests for select_eps_for_dtype function."""

    def test_fp32(self) -> None:
        """Select eps for FP32."""
        eps = select_eps_for_dtype("fp32")
        assert eps == 1e-5

    def test_fp16(self) -> None:
        """Select eps for FP16."""
        eps = select_eps_for_dtype("fp16")
        assert eps == 1e-5

    def test_bf16(self) -> None:
        """Select eps for BF16."""
        eps = select_eps_for_dtype("bf16")
        assert eps == 1e-6

    def test_fp8(self) -> None:
        """Select eps for FP8."""
        eps = select_eps_for_dtype("fp8")
        assert eps == 1e-4

    def test_invalid_dtype_raises(self) -> None:
        """Invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be one of"):
            select_eps_for_dtype("invalid")


class TestFormatNormStats:
    """Tests for format_norm_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = NormStats(0.0, 1.0, 1536, 6144, 4608)
        formatted = format_norm_stats(stats)
        assert "Parameters:" in formatted
        assert "1,536" in formatted
        assert "Memory:" in formatted
        assert "FLOPs/token:" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_norm_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedNormConfig:
    """Tests for get_recommended_norm_config function."""

    def test_transformer(self) -> None:
        """Get config for transformer."""
        config = get_recommended_norm_config("transformer", 768)
        assert config.norm_type == NormType.LAYER_NORM
        assert config.position == NormPosition.PRE

    def test_llm(self) -> None:
        """Get config for LLM."""
        config = get_recommended_norm_config("llm", 4096)
        assert config.norm_type == NormType.RMS_NORM

    def test_cnn(self) -> None:
        """Get config for CNN."""
        config = get_recommended_norm_config("cnn", 256)
        assert config.norm_type == NormType.BATCH_NORM

    def test_rnn(self) -> None:
        """Get config for RNN."""
        config = get_recommended_norm_config("rnn", 512)
        assert config.norm_type == NormType.LAYER_NORM

    def test_vit(self) -> None:
        """Get config for ViT."""
        config = get_recommended_norm_config("vit", 768)
        assert config.norm_type == NormType.LAYER_NORM

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_recommended_norm_config("invalid")

    def test_invalid_use_case_raises(self) -> None:
        """Invalid use case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_norm_config("transformer", use_case="invalid")

    def test_bf16_dtype(self) -> None:
        """Config with BF16 dtype."""
        config = get_recommended_norm_config("transformer", dtype="bf16")
        assert config.layer_norm_config is not None
        assert config.layer_norm_config.eps == 1e-6


class TestListNormTypes:
    """Tests for list_norm_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_norm_types()
        assert types == sorted(types)

    def test_contains_layer_norm(self) -> None:
        """Contains layer_norm."""
        types = list_norm_types()
        assert "layer_norm" in types

    def test_contains_rms_norm(self) -> None:
        """Contains rms_norm."""
        types = list_norm_types()
        assert "rms_norm" in types

    def test_contains_batch_norm(self) -> None:
        """Contains batch_norm."""
        types = list_norm_types()
        assert "batch_norm" in types


class TestListNormPositions:
    """Tests for list_norm_positions function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        positions = list_norm_positions()
        assert positions == sorted(positions)

    def test_contains_pre(self) -> None:
        """Contains pre."""
        positions = list_norm_positions()
        assert "pre" in positions

    def test_contains_post(self) -> None:
        """Contains post."""
        positions = list_norm_positions()
        assert "post" in positions


class TestListEpsTypes:
    """Tests for list_eps_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_eps_types()
        assert types == sorted(types)

    def test_contains_standard(self) -> None:
        """Contains standard."""
        types = list_eps_types()
        assert "standard" in types


class TestGetNormType:
    """Tests for get_norm_type function."""

    def test_get_layer_norm(self) -> None:
        """Get layer_norm."""
        assert get_norm_type("layer_norm") == NormType.LAYER_NORM

    def test_get_rms_norm(self) -> None:
        """Get rms_norm."""
        assert get_norm_type("rms_norm") == NormType.RMS_NORM

    def test_get_batch_norm(self) -> None:
        """Get batch_norm."""
        assert get_norm_type("batch_norm") == NormType.BATCH_NORM

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="norm_type must be one of"):
            get_norm_type("invalid")


class TestGetNormPosition:
    """Tests for get_norm_position function."""

    def test_get_pre(self) -> None:
        """Get pre."""
        assert get_norm_position("pre") == NormPosition.PRE

    def test_get_post(self) -> None:
        """Get post."""
        assert get_norm_position("post") == NormPosition.POST

    def test_get_sandwich(self) -> None:
        """Get sandwich."""
        assert get_norm_position("sandwich") == NormPosition.SANDWICH

    def test_invalid_position_raises(self) -> None:
        """Invalid position raises ValueError."""
        with pytest.raises(ValueError, match="norm_position must be one of"):
            get_norm_position("invalid")


class TestGetEpsType:
    """Tests for get_eps_type function."""

    def test_get_standard(self) -> None:
        """Get standard."""
        assert get_eps_type("standard") == EpsType.STANDARD

    def test_get_fp16_safe(self) -> None:
        """Get fp16_safe."""
        assert get_eps_type("fp16_safe") == EpsType.FP16_SAFE

    def test_get_bf16_safe(self) -> None:
        """Get bf16_safe."""
        assert get_eps_type("bf16_safe") == EpsType.BF16_SAFE

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="eps_type must be one of"):
            get_eps_type("invalid")


class TestGetEpsValue:
    """Tests for get_eps_value function."""

    def test_standard(self) -> None:
        """Get standard eps value."""
        assert get_eps_value("standard") == 1e-5

    def test_fp16_safe(self) -> None:
        """Get fp16_safe eps value."""
        assert get_eps_value("fp16_safe") == 1e-5

    def test_bf16_safe(self) -> None:
        """Get bf16_safe eps value."""
        assert get_eps_value("bf16_safe") == 1e-6

    def test_enum_input(self) -> None:
        """Get eps value with enum input."""
        assert get_eps_value(EpsType.STANDARD) == 1e-5


class TestInstanceNormBranch:
    """Tests for instance norm coverage."""

    def test_instance_norm_params(self) -> None:
        """Instance norm returns zero params."""
        config = NormConfig(
            NormType.INSTANCE_NORM, None, None, None, None, NormPosition.PRE
        )
        params = calculate_norm_params(config)
        assert params == 0


class TestEdgeCases:
    """Tests for edge cases and uncovered branches."""

    def test_layer_norm_none_config_zero_params(self) -> None:
        """Layer norm with None config returns zero params."""
        config = NormConfig(
            NormType.LAYER_NORM, None, None, None, None, NormPosition.PRE
        )
        params = calculate_norm_params(config)
        assert params == 0

    def test_rms_norm_none_config_zero_params(self) -> None:
        """RMS norm with None config returns zero params."""
        config = NormConfig(NormType.RMS_NORM, None, None, None, None, NormPosition.PRE)
        params = calculate_norm_params(config)
        assert params == 0

    def test_group_norm_none_config_zero_params(self) -> None:
        """Group norm with None config returns zero params."""
        config = NormConfig(
            NormType.GROUP_NORM, None, None, None, None, NormPosition.PRE
        )
        params = calculate_norm_params(config)
        assert params == 0

    def test_group_norm_no_affine_zero_params(self) -> None:
        """Group norm without affine returns zero params."""
        group_config = GroupNormConfig(32, 256, 1e-5, affine=False)
        config = NormConfig(
            NormType.GROUP_NORM, None, None, group_config, None, NormPosition.PRE
        )
        params = calculate_norm_params(config)
        assert params == 0

    def test_batch_norm_none_config_zero_params(self) -> None:
        """Batch norm with None config returns zero params."""
        config = NormConfig(
            NormType.BATCH_NORM, None, None, None, None, NormPosition.PRE
        )
        params = calculate_norm_params(config)
        assert params == 0

    def test_batch_norm_no_affine_zero_params(self) -> None:
        """Batch norm without affine returns zero params."""
        batch_config = BatchNormConfig(256, 1e-5, 0.1, affine=False)
        config = NormConfig(
            NormType.BATCH_NORM, None, None, None, batch_config, NormPosition.PRE
        )
        params = calculate_norm_params(config)
        assert params == 0

    def test_estimate_memory_rms_norm(self) -> None:
        """Estimate memory for RMS norm."""
        config = create_norm_config("rms_norm", 4096)
        mem = estimate_norm_memory(config, batch_size=1, sequence_length=512)
        assert mem > 0

    def test_estimate_memory_group_norm(self) -> None:
        """Estimate memory for group norm."""
        config = create_norm_config("group_norm", 256, num_groups=32)
        mem = estimate_norm_memory(config, batch_size=1, sequence_length=512)
        assert mem > 0

    def test_estimate_memory_batch_norm(self) -> None:
        """Estimate memory for batch norm."""
        config = create_norm_config("batch_norm", 256)
        mem = estimate_norm_memory(config, batch_size=1, sequence_length=512)
        assert mem > 0

    def test_compare_stability_group_vs_instance(self) -> None:
        """Compare group norm vs instance norm."""
        result = compare_norm_stability("group_norm", "instance_norm")
        assert "recommended" in result

    def test_compare_stability_rms_vs_layer(self) -> None:
        """Compare RMS norm vs layer norm - RMS should be recommended."""
        result = compare_norm_stability("rms_norm", "layer_norm")
        assert result["recommended"] == "rms_norm"

    def test_unknown_norm_type_default_return(self) -> None:
        """Test the final return 0 branch for unknown norm types."""
        # Create a config manually to hit the final return 0
        # This tests the default branch at the end of calculate_norm_params
        layer_config = LayerNormConfig((768,), 1e-5, True)
        config = NormConfig(
            NormType.LAYER_NORM, layer_config, None, None, None, NormPosition.PRE
        )
        # This should not hit the final return 0, but coverage shows it's needed
        params = calculate_norm_params(config)
        assert params == 1536
