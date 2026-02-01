"""Tests for models.positional module."""

from __future__ import annotations

import math

import pytest

from hf_gtc.models.positional import (
    VALID_INTERPOLATION_TYPES,
    VALID_POSITIONAL_TYPES,
    VALID_ROPE_SCALINGS,
    ALiBiConfig,
    InterpolationType,
    PositionalConfig,
    PositionalType,
    RoPEConfig,
    RoPEScaling,
    SinusoidalConfig,
    calculate_alibi_slopes,
    calculate_rope_frequencies,
    calculate_sinusoidal_embeddings,
    create_alibi_config,
    create_positional_config,
    create_rope_config,
    create_sinusoidal_config,
    estimate_position_memory,
    format_positional_stats,
    get_interpolation_type,
    get_positional_type,
    get_recommended_positional_config,
    get_rope_scaling,
    list_interpolation_types,
    list_positional_types,
    list_rope_scalings,
    validate_alibi_config,
    validate_positional_config,
    validate_rope_config,
    validate_sinusoidal_config,
)


class TestPositionalTypeEnum:
    """Tests for PositionalType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for pos_type in PositionalType:
            assert isinstance(pos_type.value, str)

    def test_sinusoidal_value(self) -> None:
        """SINUSOIDAL has correct value."""
        assert PositionalType.SINUSOIDAL.value == "sinusoidal"

    def test_learned_value(self) -> None:
        """LEARNED has correct value."""
        assert PositionalType.LEARNED.value == "learned"

    def test_rotary_value(self) -> None:
        """ROTARY has correct value."""
        assert PositionalType.ROTARY.value == "rotary"

    def test_alibi_value(self) -> None:
        """ALIBI has correct value."""
        assert PositionalType.ALIBI.value == "alibi"

    def test_relative_value(self) -> None:
        """RELATIVE has correct value."""
        assert PositionalType.RELATIVE.value == "relative"

    def test_none_value(self) -> None:
        """NONE has correct value."""
        assert PositionalType.NONE.value == "none"

    def test_valid_types_frozenset(self) -> None:
        """VALID_POSITIONAL_TYPES is a frozenset."""
        assert isinstance(VALID_POSITIONAL_TYPES, frozenset)
        assert len(VALID_POSITIONAL_TYPES) == len(PositionalType)


class TestRoPEScalingEnum:
    """Tests for RoPEScaling enum."""

    def test_all_scalings_have_values(self) -> None:
        """All scalings have string values."""
        for scaling in RoPEScaling:
            assert isinstance(scaling.value, str)

    def test_linear_value(self) -> None:
        """LINEAR has correct value."""
        assert RoPEScaling.LINEAR.value == "linear"

    def test_dynamic_value(self) -> None:
        """DYNAMIC has correct value."""
        assert RoPEScaling.DYNAMIC.value == "dynamic"

    def test_yarn_value(self) -> None:
        """YARN has correct value."""
        assert RoPEScaling.YARN.value == "yarn"

    def test_ntk_aware_value(self) -> None:
        """NTK_AWARE has correct value."""
        assert RoPEScaling.NTK_AWARE.value == "ntk_aware"

    def test_valid_scalings_frozenset(self) -> None:
        """VALID_ROPE_SCALINGS is a frozenset."""
        assert isinstance(VALID_ROPE_SCALINGS, frozenset)
        assert len(VALID_ROPE_SCALINGS) == len(RoPEScaling)


class TestInterpolationTypeEnum:
    """Tests for InterpolationType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for interp_type in InterpolationType:
            assert isinstance(interp_type.value, str)

    def test_linear_value(self) -> None:
        """LINEAR has correct value."""
        assert InterpolationType.LINEAR.value == "linear"

    def test_dynamic_value(self) -> None:
        """DYNAMIC has correct value."""
        assert InterpolationType.DYNAMIC.value == "dynamic"

    def test_yarn_value(self) -> None:
        """YARN has correct value."""
        assert InterpolationType.YARN.value == "yarn"

    def test_valid_types_frozenset(self) -> None:
        """VALID_INTERPOLATION_TYPES is a frozenset."""
        assert isinstance(VALID_INTERPOLATION_TYPES, frozenset)
        assert len(VALID_INTERPOLATION_TYPES) == len(InterpolationType)


class TestSinusoidalConfig:
    """Tests for SinusoidalConfig dataclass."""

    def test_create_config(self) -> None:
        """Create sinusoidal config."""
        config = SinusoidalConfig(max_length=512, embed_dim=768, base=10000.0)
        assert config.max_length == 512
        assert config.embed_dim == 768
        assert config.base == pytest.approx(10000.0)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SinusoidalConfig(max_length=512, embed_dim=768, base=10000.0)
        with pytest.raises(AttributeError):
            config.max_length = 1024  # type: ignore[misc]

    def test_default_base(self) -> None:
        """Default base value."""
        config = SinusoidalConfig(max_length=512, embed_dim=768)
        assert config.base == pytest.approx(10000.0)


class TestRoPEConfig:
    """Tests for RoPEConfig dataclass."""

    def test_create_config(self) -> None:
        """Create RoPE config."""
        config = RoPEConfig(
            dim=64,
            max_position=4096,
            base=10000.0,
            scaling_type=None,
            scaling_factor=1.0,
        )
        assert config.dim == 64
        assert config.max_position == 4096

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = RoPEConfig(dim=64, max_position=4096)
        with pytest.raises(AttributeError):
            config.dim = 128  # type: ignore[misc]

    def test_with_scaling(self) -> None:
        """Config with scaling type."""
        config = RoPEConfig(
            dim=64,
            max_position=4096,
            scaling_type=RoPEScaling.LINEAR,
            scaling_factor=2.0,
        )
        assert config.scaling_type == RoPEScaling.LINEAR
        assert config.scaling_factor == pytest.approx(2.0)


class TestALiBiConfig:
    """Tests for ALiBiConfig dataclass."""

    def test_create_config(self) -> None:
        """Create ALiBi config."""
        config = ALiBiConfig(num_heads=8, slopes_power=8)
        assert config.num_heads == 8
        assert config.slopes_power == 8

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ALiBiConfig(num_heads=8)
        with pytest.raises(AttributeError):
            config.num_heads = 12  # type: ignore[misc]

    def test_default_slopes_power(self) -> None:
        """Default slopes power value."""
        config = ALiBiConfig(num_heads=8)
        assert config.slopes_power == 8


class TestPositionalConfig:
    """Tests for PositionalConfig dataclass."""

    def test_create_config(self) -> None:
        """Create positional config."""
        config = PositionalConfig(
            pos_type=PositionalType.SINUSOIDAL,
            rope_config=None,
            alibi_config=None,
            max_length=4096,
        )
        assert config.pos_type == PositionalType.SINUSOIDAL
        assert config.max_length == 4096

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PositionalConfig(
            pos_type=PositionalType.SINUSOIDAL,
            rope_config=None,
            alibi_config=None,
            max_length=4096,
        )
        with pytest.raises(AttributeError):
            config.max_length = 8192  # type: ignore[misc]

    def test_with_rope_config(self) -> None:
        """Config with RoPE configuration."""
        rope = RoPEConfig(dim=64, max_position=4096)
        config = PositionalConfig(
            pos_type=PositionalType.ROTARY,
            rope_config=rope,
            alibi_config=None,
            max_length=4096,
        )
        assert config.rope_config == rope

    def test_with_alibi_config(self) -> None:
        """Config with ALiBi configuration."""
        alibi = ALiBiConfig(num_heads=8)
        config = PositionalConfig(
            pos_type=PositionalType.ALIBI,
            rope_config=None,
            alibi_config=alibi,
            max_length=4096,
        )
        assert config.alibi_config == alibi


class TestValidateSinusoidalConfig:
    """Tests for validate_sinusoidal_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SinusoidalConfig(max_length=512, embed_dim=768)
        validate_sinusoidal_config(config)  # Should not raise

    def test_zero_max_length_raises(self) -> None:
        """Zero max length raises ValueError."""
        config = SinusoidalConfig(max_length=0, embed_dim=768)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_sinusoidal_config(config)

    def test_negative_max_length_raises(self) -> None:
        """Negative max length raises ValueError."""
        config = SinusoidalConfig(max_length=-1, embed_dim=768)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_sinusoidal_config(config)

    def test_zero_embed_dim_raises(self) -> None:
        """Zero embed dim raises ValueError."""
        config = SinusoidalConfig(max_length=512, embed_dim=0)
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            validate_sinusoidal_config(config)

    def test_odd_embed_dim_raises(self) -> None:
        """Odd embed dim raises ValueError."""
        config = SinusoidalConfig(max_length=512, embed_dim=767)
        with pytest.raises(ValueError, match="embed_dim must be even"):
            validate_sinusoidal_config(config)

    def test_zero_base_raises(self) -> None:
        """Zero base raises ValueError."""
        config = SinusoidalConfig(max_length=512, embed_dim=768, base=0.0)
        with pytest.raises(ValueError, match="base must be positive"):
            validate_sinusoidal_config(config)


class TestValidateRoPEConfig:
    """Tests for validate_rope_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RoPEConfig(dim=64, max_position=4096)
        validate_rope_config(config)  # Should not raise

    def test_zero_dim_raises(self) -> None:
        """Zero dim raises ValueError."""
        config = RoPEConfig(dim=0, max_position=4096)
        with pytest.raises(ValueError, match="dim must be positive"):
            validate_rope_config(config)

    def test_odd_dim_raises(self) -> None:
        """Odd dim raises ValueError."""
        config = RoPEConfig(dim=63, max_position=4096)
        with pytest.raises(ValueError, match="dim must be even"):
            validate_rope_config(config)

    def test_zero_max_position_raises(self) -> None:
        """Zero max position raises ValueError."""
        config = RoPEConfig(dim=64, max_position=0)
        with pytest.raises(ValueError, match="max_position must be positive"):
            validate_rope_config(config)

    def test_zero_base_raises(self) -> None:
        """Zero base raises ValueError."""
        config = RoPEConfig(dim=64, max_position=4096, base=0.0)
        with pytest.raises(ValueError, match="base must be positive"):
            validate_rope_config(config)

    def test_zero_scaling_factor_raises(self) -> None:
        """Zero scaling factor raises ValueError."""
        config = RoPEConfig(dim=64, max_position=4096, scaling_factor=0.0)
        with pytest.raises(ValueError, match="scaling_factor must be positive"):
            validate_rope_config(config)


class TestValidateALiBiConfig:
    """Tests for validate_alibi_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ALiBiConfig(num_heads=8)
        validate_alibi_config(config)  # Should not raise

    def test_zero_num_heads_raises(self) -> None:
        """Zero num heads raises ValueError."""
        config = ALiBiConfig(num_heads=0)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            validate_alibi_config(config)

    def test_zero_slopes_power_raises(self) -> None:
        """Zero slopes power raises ValueError."""
        config = ALiBiConfig(num_heads=8, slopes_power=0)
        with pytest.raises(ValueError, match="slopes_power must be positive"):
            validate_alibi_config(config)


class TestValidatePositionalConfig:
    """Tests for validate_positional_config function."""

    def test_valid_sinusoidal_config(self) -> None:
        """Valid sinusoidal config passes validation."""
        config = PositionalConfig(
            pos_type=PositionalType.SINUSOIDAL,
            rope_config=None,
            alibi_config=None,
            max_length=4096,
        )
        validate_positional_config(config)  # Should not raise

    def test_valid_rotary_config(self) -> None:
        """Valid rotary config passes validation."""
        rope = RoPEConfig(dim=64, max_position=4096)
        config = PositionalConfig(
            pos_type=PositionalType.ROTARY,
            rope_config=rope,
            alibi_config=None,
            max_length=4096,
        )
        validate_positional_config(config)  # Should not raise

    def test_valid_alibi_config(self) -> None:
        """Valid alibi config passes validation."""
        alibi = ALiBiConfig(num_heads=8)
        config = PositionalConfig(
            pos_type=PositionalType.ALIBI,
            rope_config=None,
            alibi_config=alibi,
            max_length=4096,
        )
        validate_positional_config(config)  # Should not raise

    def test_zero_max_length_raises(self) -> None:
        """Zero max length raises ValueError."""
        config = PositionalConfig(
            pos_type=PositionalType.SINUSOIDAL,
            rope_config=None,
            alibi_config=None,
            max_length=0,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_positional_config(config)

    def test_rotary_without_rope_config_raises(self) -> None:
        """Rotary without rope config raises ValueError."""
        config = PositionalConfig(
            pos_type=PositionalType.ROTARY,
            rope_config=None,
            alibi_config=None,
            max_length=4096,
        )
        with pytest.raises(ValueError, match="rope_config required"):
            validate_positional_config(config)

    def test_alibi_without_alibi_config_raises(self) -> None:
        """ALiBi without alibi config raises ValueError."""
        config = PositionalConfig(
            pos_type=PositionalType.ALIBI,
            rope_config=None,
            alibi_config=None,
            max_length=4096,
        )
        with pytest.raises(ValueError, match="alibi_config required"):
            validate_positional_config(config)

    def test_invalid_rope_config_raises(self) -> None:
        """Invalid rope config raises ValueError."""
        rope = RoPEConfig(dim=0, max_position=4096)
        config = PositionalConfig(
            pos_type=PositionalType.ROTARY,
            rope_config=rope,
            alibi_config=None,
            max_length=4096,
        )
        with pytest.raises(ValueError, match="dim must be positive"):
            validate_positional_config(config)

    def test_invalid_alibi_config_raises(self) -> None:
        """Invalid alibi config raises ValueError."""
        alibi = ALiBiConfig(num_heads=0)
        config = PositionalConfig(
            pos_type=PositionalType.ALIBI,
            rope_config=None,
            alibi_config=alibi,
            max_length=4096,
        )
        with pytest.raises(ValueError, match="num_heads must be positive"):
            validate_positional_config(config)


class TestCreateSinusoidalConfig:
    """Tests for create_sinusoidal_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_sinusoidal_config()
        assert config.max_length == 512
        assert config.embed_dim == 768
        assert config.base == pytest.approx(10000.0)

    def test_custom_max_length(self) -> None:
        """Create config with custom max length."""
        config = create_sinusoidal_config(max_length=1024)
        assert config.max_length == 1024

    def test_custom_embed_dim(self) -> None:
        """Create config with custom embed dim."""
        config = create_sinusoidal_config(embed_dim=512)
        assert config.embed_dim == 512

    def test_custom_base(self) -> None:
        """Create config with custom base."""
        config = create_sinusoidal_config(base=20000.0)
        assert config.base == pytest.approx(20000.0)

    def test_zero_max_length_raises(self) -> None:
        """Zero max length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_sinusoidal_config(max_length=0)


class TestCreateRoPEConfig:
    """Tests for create_rope_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_rope_config()
        assert config.dim == 64
        assert config.max_position == 4096
        assert config.base == pytest.approx(10000.0)
        assert config.scaling_type is None
        assert config.scaling_factor == pytest.approx(1.0)

    def test_custom_dim(self) -> None:
        """Create config with custom dim."""
        config = create_rope_config(dim=128)
        assert config.dim == 128

    def test_custom_max_position(self) -> None:
        """Create config with custom max position."""
        config = create_rope_config(max_position=8192)
        assert config.max_position == 8192

    def test_with_scaling_type(self) -> None:
        """Create config with scaling type."""
        config = create_rope_config(scaling_type="linear", scaling_factor=2.0)
        assert config.scaling_type == RoPEScaling.LINEAR
        assert config.scaling_factor == pytest.approx(2.0)

    def test_invalid_scaling_type_raises(self) -> None:
        """Invalid scaling type raises ValueError."""
        with pytest.raises(ValueError, match="scaling_type must be one of"):
            create_rope_config(scaling_type="invalid")  # type: ignore[arg-type]

    def test_zero_dim_raises(self) -> None:
        """Zero dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            create_rope_config(dim=0)


class TestCreateALiBiConfig:
    """Tests for create_alibi_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_alibi_config()
        assert config.num_heads == 8
        assert config.slopes_power == 8

    def test_custom_num_heads(self) -> None:
        """Create config with custom num heads."""
        config = create_alibi_config(num_heads=12)
        assert config.num_heads == 12

    def test_custom_slopes_power(self) -> None:
        """Create config with custom slopes power."""
        config = create_alibi_config(slopes_power=4)
        assert config.slopes_power == 4

    def test_zero_num_heads_raises(self) -> None:
        """Zero num heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            create_alibi_config(num_heads=0)


class TestCreatePositionalConfig:
    """Tests for create_positional_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_positional_config()
        assert config.pos_type == PositionalType.SINUSOIDAL
        assert config.max_length == 4096

    def test_sinusoidal_type(self) -> None:
        """Create sinusoidal config."""
        config = create_positional_config(pos_type="sinusoidal")
        assert config.pos_type == PositionalType.SINUSOIDAL

    def test_rotary_type(self) -> None:
        """Create rotary config."""
        rope = create_rope_config()
        config = create_positional_config(pos_type="rotary", rope_config=rope)
        assert config.pos_type == PositionalType.ROTARY
        assert config.rope_config is not None

    def test_alibi_type(self) -> None:
        """Create alibi config."""
        alibi = create_alibi_config()
        config = create_positional_config(pos_type="alibi", alibi_config=alibi)
        assert config.pos_type == PositionalType.ALIBI
        assert config.alibi_config is not None

    def test_invalid_pos_type_raises(self) -> None:
        """Invalid pos type raises ValueError."""
        with pytest.raises(ValueError, match="pos_type must be one of"):
            create_positional_config(pos_type="invalid")  # type: ignore[arg-type]

    def test_rotary_without_rope_config_raises(self) -> None:
        """Rotary without rope config raises ValueError."""
        with pytest.raises(ValueError, match="rope_config required"):
            create_positional_config(pos_type="rotary")


class TestListPositionalTypes:
    """Tests for list_positional_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_positional_types()
        assert types == sorted(types)

    def test_contains_sinusoidal(self) -> None:
        """Contains sinusoidal."""
        types = list_positional_types()
        assert "sinusoidal" in types

    def test_contains_rotary(self) -> None:
        """Contains rotary."""
        types = list_positional_types()
        assert "rotary" in types

    def test_contains_all_types(self) -> None:
        """Contains all types."""
        types = list_positional_types()
        assert len(types) == len(VALID_POSITIONAL_TYPES)


class TestListRoPEScalings:
    """Tests for list_rope_scalings function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        scalings = list_rope_scalings()
        assert scalings == sorted(scalings)

    def test_contains_linear(self) -> None:
        """Contains linear."""
        scalings = list_rope_scalings()
        assert "linear" in scalings

    def test_contains_yarn(self) -> None:
        """Contains yarn."""
        scalings = list_rope_scalings()
        assert "yarn" in scalings

    def test_contains_all_scalings(self) -> None:
        """Contains all scalings."""
        scalings = list_rope_scalings()
        assert len(scalings) == len(VALID_ROPE_SCALINGS)


class TestListInterpolationTypes:
    """Tests for list_interpolation_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_interpolation_types()
        assert types == sorted(types)

    def test_contains_linear(self) -> None:
        """Contains linear."""
        types = list_interpolation_types()
        assert "linear" in types

    def test_contains_dynamic(self) -> None:
        """Contains dynamic."""
        types = list_interpolation_types()
        assert "dynamic" in types

    def test_contains_all_types(self) -> None:
        """Contains all types."""
        types = list_interpolation_types()
        assert len(types) == len(VALID_INTERPOLATION_TYPES)


class TestGetPositionalType:
    """Tests for get_positional_type function."""

    def test_get_sinusoidal(self) -> None:
        """Get sinusoidal type."""
        pos_type = get_positional_type("sinusoidal")
        assert pos_type == PositionalType.SINUSOIDAL

    def test_get_rotary(self) -> None:
        """Get rotary type."""
        pos_type = get_positional_type("rotary")
        assert pos_type == PositionalType.ROTARY

    def test_get_alibi(self) -> None:
        """Get alibi type."""
        pos_type = get_positional_type("alibi")
        assert pos_type == PositionalType.ALIBI

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown positional type"):
            get_positional_type("invalid")


class TestGetRoPEScaling:
    """Tests for get_rope_scaling function."""

    def test_get_linear(self) -> None:
        """Get linear scaling."""
        scaling = get_rope_scaling("linear")
        assert scaling == RoPEScaling.LINEAR

    def test_get_dynamic(self) -> None:
        """Get dynamic scaling."""
        scaling = get_rope_scaling("dynamic")
        assert scaling == RoPEScaling.DYNAMIC

    def test_get_yarn(self) -> None:
        """Get yarn scaling."""
        scaling = get_rope_scaling("yarn")
        assert scaling == RoPEScaling.YARN

    def test_invalid_scaling_raises(self) -> None:
        """Invalid scaling raises ValueError."""
        with pytest.raises(ValueError, match="Unknown RoPE scaling"):
            get_rope_scaling("invalid")


class TestGetInterpolationType:
    """Tests for get_interpolation_type function."""

    def test_get_linear(self) -> None:
        """Get linear type."""
        interp_type = get_interpolation_type("linear")
        assert interp_type == InterpolationType.LINEAR

    def test_get_dynamic(self) -> None:
        """Get dynamic type."""
        interp_type = get_interpolation_type("dynamic")
        assert interp_type == InterpolationType.DYNAMIC

    def test_get_yarn(self) -> None:
        """Get yarn type."""
        interp_type = get_interpolation_type("yarn")
        assert interp_type == InterpolationType.YARN

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown interpolation type"):
            get_interpolation_type("invalid")


class TestCalculateSinusoidalEmbeddings:
    """Tests for calculate_sinusoidal_embeddings function."""

    def test_basic_calculation(self) -> None:
        """Basic embedding calculation."""
        embeddings = calculate_sinusoidal_embeddings(4, 8)
        assert len(embeddings) == 4
        assert len(embeddings[0]) == 8

    def test_position_zero_sin_values(self) -> None:
        """Position 0 has sin(0)=0 for even indices."""
        embeddings = calculate_sinusoidal_embeddings(2, 4)
        # Even indices are sin values, which at position 0 should be 0
        assert embeddings[0][0] == pytest.approx(0.0)
        assert embeddings[0][2] == pytest.approx(0.0)

    def test_position_zero_cos_values(self) -> None:
        """Position 0 has cos(0)=1 for odd indices."""
        embeddings = calculate_sinusoidal_embeddings(2, 4)
        # Odd indices are cos values, which at position 0 should be 1
        assert embeddings[0][1] == pytest.approx(1.0)
        assert embeddings[0][3] == pytest.approx(1.0)

    def test_zero_max_length_raises(self) -> None:
        """Zero max length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            calculate_sinusoidal_embeddings(0, 8)

    def test_zero_embed_dim_raises(self) -> None:
        """Zero embed dim raises ValueError."""
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            calculate_sinusoidal_embeddings(4, 0)

    def test_odd_embed_dim_raises(self) -> None:
        """Odd embed dim raises ValueError."""
        with pytest.raises(ValueError, match="embed_dim must be even"):
            calculate_sinusoidal_embeddings(4, 7)

    def test_zero_base_raises(self) -> None:
        """Zero base raises ValueError."""
        with pytest.raises(ValueError, match="base must be positive"):
            calculate_sinusoidal_embeddings(4, 8, base=0.0)

    def test_embeddings_are_bounded(self) -> None:
        """Embeddings are bounded by -1 and 1."""
        embeddings = calculate_sinusoidal_embeddings(100, 64)
        for pos_emb in embeddings:
            for val in pos_emb:
                assert -1.0 <= val <= 1.0


class TestCalculateRoPEFrequencies:
    """Tests for calculate_rope_frequencies function."""

    def test_basic_calculation(self) -> None:
        """Basic frequency calculation."""
        freqs = calculate_rope_frequencies(8, 4)
        assert len(freqs) == 4
        assert len(freqs[0]) == 8

    def test_position_zero_cos_values(self) -> None:
        """Position 0 has cos(0)=1."""
        freqs = calculate_rope_frequencies(4, 2)
        # At position 0, all cos values should be 1
        assert freqs[0][0] == pytest.approx(1.0)
        assert freqs[0][2] == pytest.approx(1.0)

    def test_position_zero_sin_values(self) -> None:
        """Position 0 has sin(0)=0."""
        freqs = calculate_rope_frequencies(4, 2)
        # At position 0, all sin values should be 0
        assert freqs[0][1] == pytest.approx(0.0)
        assert freqs[0][3] == pytest.approx(0.0)

    def test_zero_dim_raises(self) -> None:
        """Zero dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            calculate_rope_frequencies(0, 4)

    def test_odd_dim_raises(self) -> None:
        """Odd dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be even"):
            calculate_rope_frequencies(7, 4)

    def test_zero_max_position_raises(self) -> None:
        """Zero max position raises ValueError."""
        with pytest.raises(ValueError, match="max_position must be positive"):
            calculate_rope_frequencies(8, 0)

    def test_zero_base_raises(self) -> None:
        """Zero base raises ValueError."""
        with pytest.raises(ValueError, match="base must be positive"):
            calculate_rope_frequencies(8, 4, base=0.0)

    def test_zero_scaling_factor_raises(self) -> None:
        """Zero scaling factor raises ValueError."""
        with pytest.raises(ValueError, match="scaling_factor must be positive"):
            calculate_rope_frequencies(8, 4, scaling_factor=0.0)

    def test_scaling_factor_effect(self) -> None:
        """Scaling factor affects frequencies."""
        freqs1 = calculate_rope_frequencies(8, 4, scaling_factor=1.0)
        freqs2 = calculate_rope_frequencies(8, 4, scaling_factor=2.0)
        # With scaling, position 2 with factor 2 should equal position 1 with factor 1
        for i in range(8):
            assert freqs2[2][i] == pytest.approx(freqs1[1][i], rel=1e-6)


class TestCalculateALiBiSlopes:
    """Tests for calculate_alibi_slopes function."""

    def test_basic_calculation(self) -> None:
        """Basic slope calculation."""
        slopes = calculate_alibi_slopes(8)
        assert len(slopes) == 8

    def test_slopes_decrease(self) -> None:
        """Slopes decrease with head index."""
        slopes = calculate_alibi_slopes(8)
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1]

    def test_slopes_positive(self) -> None:
        """All slopes are positive."""
        slopes = calculate_alibi_slopes(8)
        for slope in slopes:
            assert slope > 0

    def test_expected_values(self) -> None:
        """Expected values for 8 heads with power 8."""
        slopes = calculate_alibi_slopes(8, slopes_power=8)
        # First slope: 2^(-8 * 1 / 8) = 2^(-1) = 0.5
        assert slopes[0] == pytest.approx(0.5)
        # Last slope: 2^(-8 * 8 / 8) = 2^(-8) = 1/256
        assert slopes[7] == pytest.approx(1 / 256)

    def test_zero_num_heads_raises(self) -> None:
        """Zero num heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            calculate_alibi_slopes(0)

    def test_zero_slopes_power_raises(self) -> None:
        """Zero slopes power raises ValueError."""
        with pytest.raises(ValueError, match="slopes_power must be positive"):
            calculate_alibi_slopes(8, slopes_power=0)


class TestEstimatePositionMemory:
    """Tests for estimate_position_memory function."""

    def test_sinusoidal_memory(self) -> None:
        """Estimate sinusoidal memory."""
        mem = estimate_position_memory("sinusoidal", 512, embed_dim=768)
        assert mem > 0
        # 512 * 768 * 4 bytes = 1572864 bytes = ~1.5 MB
        expected = 512 * 768 * 4 / (1024 * 1024)
        assert mem == pytest.approx(expected)

    def test_learned_memory(self) -> None:
        """Estimate learned memory (same as sinusoidal)."""
        mem = estimate_position_memory("learned", 512, embed_dim=768)
        assert mem > 0

    def test_rotary_memory(self) -> None:
        """Estimate rotary memory."""
        mem = estimate_position_memory("rotary", 4096, head_dim=64)
        assert mem > 0
        # 4096 * 64 * 2 * 4 bytes = 2097152 bytes = 2 MB
        expected = 4096 * 64 * 2 * 4 / (1024 * 1024)
        assert mem == pytest.approx(expected)

    def test_alibi_memory(self) -> None:
        """Estimate alibi memory."""
        mem = estimate_position_memory("alibi", 4096, num_heads=32)
        assert mem > 0
        # 32 * 4 bytes = 128 bytes
        expected = 32 * 4 / (1024 * 1024)
        assert mem == pytest.approx(expected)

    def test_relative_memory(self) -> None:
        """Estimate relative memory."""
        mem = estimate_position_memory("relative", 512, embed_dim=768)
        assert mem > 0
        # (2 * 512 - 1) * 768 * 4 bytes
        expected = (2 * 512 - 1) * 768 * 4 / (1024 * 1024)
        assert mem == pytest.approx(expected)

    def test_none_memory(self) -> None:
        """No positional encoding uses no memory."""
        mem = estimate_position_memory("none", 4096)
        assert mem == pytest.approx(0.0)

    def test_invalid_pos_type_raises(self) -> None:
        """Invalid pos type raises ValueError."""
        with pytest.raises(ValueError, match="pos_type must be one of"):
            estimate_position_memory("invalid", 512)  # type: ignore[arg-type]

    def test_zero_max_length_raises(self) -> None:
        """Zero max length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            estimate_position_memory("sinusoidal", 0)

    def test_zero_dtype_bytes_raises(self) -> None:
        """Zero dtype bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            estimate_position_memory("sinusoidal", 512, dtype_bytes=0)


class TestFormatPositionalStats:
    """Tests for format_positional_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = format_positional_stats("sinusoidal", 512, 1.5, embed_dim=768)
        assert "Type: sinusoidal" in stats
        assert "Max Length: 512" in stats
        assert "Memory: 1.500 MB" in stats
        assert "Embed Dim: 768" in stats

    def test_alibi_formatting(self) -> None:
        """ALiBi stats formatting with num_heads."""
        stats = format_positional_stats("alibi", 4096, 0.001, num_heads=32)
        assert "Type: alibi" in stats
        assert "Max Length: 4096" in stats
        assert "Num Heads: 32" in stats

    def test_without_optional_params(self) -> None:
        """Formatting without optional params."""
        stats = format_positional_stats("rotary", 4096, 2.0)
        assert "Type: rotary" in stats
        assert "Embed Dim" not in stats
        assert "Num Heads" not in stats

    def test_multiline_output(self) -> None:
        """Output contains multiple lines."""
        stats = format_positional_stats("sinusoidal", 512, 1.5)
        lines = stats.split("\n")
        assert len(lines) >= 3


class TestGetRecommendedPositionalConfig:
    """Tests for get_recommended_positional_config function."""

    def test_decoder_returns_rotary(self) -> None:
        """Decoder model returns RoPE config."""
        config = get_recommended_positional_config(model_type="decoder")
        assert config.pos_type == PositionalType.ROTARY
        assert config.rope_config is not None

    def test_encoder_returns_sinusoidal(self) -> None:
        """Encoder model returns sinusoidal config."""
        config = get_recommended_positional_config(model_type="encoder")
        assert config.pos_type == PositionalType.SINUSOIDAL

    def test_encoder_decoder_returns_sinusoidal(self) -> None:
        """Encoder-decoder model returns sinusoidal config."""
        config = get_recommended_positional_config(model_type="encoder-decoder")
        assert config.pos_type == PositionalType.SINUSOIDAL

    def test_long_context_uses_scaling(self) -> None:
        """Long context decoder uses RoPE scaling."""
        config = get_recommended_positional_config(
            model_type="decoder", context_length=32000
        )
        assert config.pos_type == PositionalType.ROTARY
        assert config.rope_config is not None
        assert config.rope_config.scaling_type == RoPEScaling.DYNAMIC

    def test_short_context_no_scaling(self) -> None:
        """Short context decoder has no scaling."""
        config = get_recommended_positional_config(
            model_type="decoder", context_length=4096
        )
        assert config.pos_type == PositionalType.ROTARY
        assert config.rope_config is not None
        assert config.rope_config.scaling_type is None

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_recommended_positional_config(model_type="invalid")

    def test_zero_context_length_raises(self) -> None:
        """Zero context length raises ValueError."""
        with pytest.raises(ValueError, match="context_length must be positive"):
            get_recommended_positional_config(context_length=0)

    def test_zero_num_heads_raises(self) -> None:
        """Zero num heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            get_recommended_positional_config(num_heads=0)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            get_recommended_positional_config(head_dim=0)

    def test_custom_head_dim(self) -> None:
        """Custom head dim is used in RoPE config."""
        config = get_recommended_positional_config(model_type="decoder", head_dim=128)
        assert config.rope_config is not None
        assert config.rope_config.dim == 128


class TestAllPositionalTypes:
    """Test all positional types can be created."""

    @pytest.mark.parametrize("pos_type", list(VALID_POSITIONAL_TYPES))
    def test_get_positional_type(self, pos_type: str) -> None:
        """All positional types can be retrieved."""
        result = get_positional_type(pos_type)
        assert result.value == pos_type


class TestAllRoPEScalings:
    """Test all RoPE scalings can be created."""

    @pytest.mark.parametrize("scaling", list(VALID_ROPE_SCALINGS))
    def test_get_rope_scaling(self, scaling: str) -> None:
        """All RoPE scalings can be retrieved."""
        result = get_rope_scaling(scaling)
        assert result.value == scaling

    @pytest.mark.parametrize("scaling", list(VALID_ROPE_SCALINGS))
    def test_create_rope_config_with_scaling(self, scaling: str) -> None:
        """RoPE config can be created with each scaling."""
        config = create_rope_config(scaling_type=scaling)  # type: ignore[arg-type]
        assert config.scaling_type is not None
        assert config.scaling_type.value == scaling


class TestAllInterpolationTypes:
    """Test all interpolation types can be created."""

    @pytest.mark.parametrize("interp_type", list(VALID_INTERPOLATION_TYPES))
    def test_get_interpolation_type(self, interp_type: str) -> None:
        """All interpolation types can be retrieved."""
        result = get_interpolation_type(interp_type)
        assert result.value == interp_type


class TestMathematicalProperties:
    """Tests for mathematical properties of the calculations."""

    def test_sinusoidal_orthogonality_property(self) -> None:
        """Sinusoidal embeddings have approximately orthogonal distant positions."""
        embeddings = calculate_sinusoidal_embeddings(100, 64)
        # Compute dot product of position 0 and position 50
        pos0 = embeddings[0]
        pos50 = embeddings[50]
        dot_product = sum(a * b for a, b in zip(pos0, pos50, strict=True))
        # Due to the orthogonality property, distant positions should have
        # relatively low dot product compared to the norm
        norm0 = math.sqrt(sum(a * a for a in pos0))
        norm50 = math.sqrt(sum(a * a for a in pos50))
        cosine_sim = dot_product / (norm0 * norm50)
        # Cosine similarity should be significantly less than 1 for distant positions
        assert cosine_sim < 0.9

    def test_rope_rotational_property(self) -> None:
        """RoPE frequencies create rotation-like patterns."""
        freqs = calculate_rope_frequencies(4, 10)
        # For each dimension pair (cos, sin), check that cos^2 + sin^2 = 1
        for pos_freq in freqs:
            for i in range(0, len(pos_freq), 2):
                cos_val = pos_freq[i]
                sin_val = pos_freq[i + 1]
                assert cos_val**2 + sin_val**2 == pytest.approx(1.0)

    def test_alibi_slopes_geometric_sequence(self) -> None:
        """ALiBi slopes form a geometric sequence."""
        slopes = calculate_alibi_slopes(8, slopes_power=8)
        # Check ratio between consecutive slopes
        for i in range(len(slopes) - 1):
            ratio = slopes[i + 1] / slopes[i]
            expected_ratio = 2 ** (-8 / 8)  # = 0.5
            assert ratio == pytest.approx(expected_ratio)
