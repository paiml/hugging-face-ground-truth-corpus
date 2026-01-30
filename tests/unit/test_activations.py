"""Tests for activation functions module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.models.activations import (
    VALID_ACTIVATION_TYPES,
    VALID_GELU_APPROXIMATIONS,
    VALID_GLU_VARIANTS,
    ActivationConfig,
    ActivationStats,
    ActivationType,
    GELUApproximation,
    GELUConfig,
    GLUVariant,
    SwiGLUConfig,
    calculate_activation_memory,
    calculate_gradient_magnitude,
    compare_activation_properties,
    create_activation_config,
    create_activation_stats,
    create_gelu_config,
    create_swiglu_config,
    estimate_glu_expansion,
    format_activation_stats,
    get_activation_type,
    get_gelu_approximation,
    get_glu_variant,
    get_recommended_activation_config,
    list_activation_types,
    list_gelu_approximations,
    list_glu_variants,
    validate_activation_config,
    validate_activation_stats,
    validate_gelu_config,
    validate_swiglu_config,
)


class TestActivationType:
    """Tests for ActivationType enum."""

    def test_relu_value(self) -> None:
        """Test RELU enum value."""
        assert ActivationType.RELU.value == "relu"

    def test_gelu_value(self) -> None:
        """Test GELU enum value."""
        assert ActivationType.GELU.value == "gelu"

    def test_gelu_new_value(self) -> None:
        """Test GELU_NEW enum value."""
        assert ActivationType.GELU_NEW.value == "gelu_new"

    def test_silu_value(self) -> None:
        """Test SILU enum value."""
        assert ActivationType.SILU.value == "silu"

    def test_swiglu_value(self) -> None:
        """Test SWIGLU enum value."""
        assert ActivationType.SWIGLU.value == "swiglu"

    def test_geglu_value(self) -> None:
        """Test GEGLU enum value."""
        assert ActivationType.GEGLU.value == "geglu"

    def test_mish_value(self) -> None:
        """Test MISH enum value."""
        assert ActivationType.MISH.value == "mish"

    def test_tanh_value(self) -> None:
        """Test TANH enum value."""
        assert ActivationType.TANH.value == "tanh"

    def test_sigmoid_value(self) -> None:
        """Test SIGMOID enum value."""
        assert ActivationType.SIGMOID.value == "sigmoid"

    def test_valid_activation_types_contains_all(self) -> None:
        """Test that VALID_ACTIVATION_TYPES contains all enum values."""
        for at in ActivationType:
            assert at.value in VALID_ACTIVATION_TYPES


class TestGELUApproximation:
    """Tests for GELUApproximation enum."""

    def test_none_value(self) -> None:
        """Test NONE enum value."""
        assert GELUApproximation.NONE.value == "none"

    def test_tanh_value(self) -> None:
        """Test TANH enum value."""
        assert GELUApproximation.TANH.value == "tanh"

    def test_sigmoid_value(self) -> None:
        """Test SIGMOID enum value."""
        assert GELUApproximation.SIGMOID.value == "sigmoid"

    def test_valid_gelu_approximations_contains_all(self) -> None:
        """Test that VALID_GELU_APPROXIMATIONS contains all enum values."""
        for ga in GELUApproximation:
            assert ga.value in VALID_GELU_APPROXIMATIONS


class TestGLUVariant:
    """Tests for GLUVariant enum."""

    def test_swiglu_value(self) -> None:
        """Test SWIGLU enum value."""
        assert GLUVariant.SWIGLU.value == "swiglu"

    def test_geglu_value(self) -> None:
        """Test GEGLU enum value."""
        assert GLUVariant.GEGLU.value == "geglu"

    def test_reglu_value(self) -> None:
        """Test REGLU enum value."""
        assert GLUVariant.REGLU.value == "reglu"

    def test_bilinear_value(self) -> None:
        """Test BILINEAR enum value."""
        assert GLUVariant.BILINEAR.value == "bilinear"

    def test_valid_glu_variants_contains_all(self) -> None:
        """Test that VALID_GLU_VARIANTS contains all enum values."""
        for gv in GLUVariant:
            assert gv.value in VALID_GLU_VARIANTS


class TestGELUConfig:
    """Tests for GELUConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        config = GELUConfig(approximate=GELUApproximation.NONE)
        assert config.approximate == GELUApproximation.NONE

    def test_tanh_approximation(self) -> None:
        """Test tanh approximation config."""
        config = GELUConfig(approximate=GELUApproximation.TANH)
        assert config.approximate == GELUApproximation.TANH

    def test_frozen(self) -> None:
        """Test that GELUConfig is immutable."""
        config = GELUConfig(approximate=GELUApproximation.NONE)
        with pytest.raises(AttributeError):
            config.approximate = GELUApproximation.TANH  # type: ignore[misc]


class TestSwiGLUConfig:
    """Tests for SwiGLUConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        config = SwiGLUConfig(hidden_dim=4096, bias=False, gate_dim=4096)
        assert config.hidden_dim == 4096
        assert config.bias is False
        assert config.gate_dim == 4096

    def test_with_bias(self) -> None:
        """Test creation with bias."""
        config = SwiGLUConfig(hidden_dim=2048, bias=True, gate_dim=2048)
        assert config.bias is True

    def test_different_gate_dim(self) -> None:
        """Test creation with different gate dim."""
        config = SwiGLUConfig(hidden_dim=4096, bias=False, gate_dim=2048)
        assert config.gate_dim == 2048

    def test_frozen(self) -> None:
        """Test that SwiGLUConfig is immutable."""
        config = SwiGLUConfig(hidden_dim=4096, bias=False, gate_dim=4096)
        with pytest.raises(AttributeError):
            config.hidden_dim = 2048  # type: ignore[misc]


class TestActivationConfig:
    """Tests for ActivationConfig dataclass."""

    def test_gelu_config(self) -> None:
        """Test GELU activation config."""
        config = create_activation_config(activation_type="gelu")
        assert config.activation_type == ActivationType.GELU
        assert config.gelu_config is not None
        assert config.swiglu_config is None

    def test_swiglu_config(self) -> None:
        """Test SwiGLU activation config."""
        config = create_activation_config(activation_type="swiglu", hidden_dim=2048)
        assert config.activation_type == ActivationType.SWIGLU
        assert config.swiglu_config is not None
        assert config.swiglu_config.hidden_dim == 2048

    def test_relu_config(self) -> None:
        """Test ReLU activation config."""
        config = create_activation_config(activation_type="relu")
        assert config.activation_type == ActivationType.RELU
        assert config.gelu_config is None
        assert config.swiglu_config is None

    def test_frozen(self) -> None:
        """Test that ActivationConfig is immutable."""
        config = create_activation_config(activation_type="gelu")
        with pytest.raises(AttributeError):
            config.inplace = True  # type: ignore[misc]


class TestActivationStats:
    """Tests for ActivationStats dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        stats = ActivationStats(
            memory_overhead=1.0,
            compute_cost=2.0,
            gradient_stability=0.9,
        )
        assert stats.memory_overhead == pytest.approx(1.0)
        assert stats.compute_cost == pytest.approx(2.0)
        assert stats.gradient_stability == pytest.approx(0.9)

    def test_frozen(self) -> None:
        """Test that ActivationStats is immutable."""
        stats = ActivationStats(
            memory_overhead=1.0,
            compute_cost=2.0,
            gradient_stability=0.9,
        )
        with pytest.raises(AttributeError):
            stats.memory_overhead = 2.0  # type: ignore[misc]


class TestValidateGELUConfig:
    """Tests for validate_gelu_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = GELUConfig(approximate=GELUApproximation.NONE)
        validate_gelu_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_gelu_config(None)  # type: ignore[arg-type]


class TestValidateSwiGLUConfig:
    """Tests for validate_swiglu_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = SwiGLUConfig(hidden_dim=4096, bias=False, gate_dim=4096)
        validate_swiglu_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_swiglu_config(None)  # type: ignore[arg-type]

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        config = SwiGLUConfig(hidden_dim=0, bias=False, gate_dim=4096)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_swiglu_config(config)

    def test_negative_hidden_dim_raises_error(self) -> None:
        """Test that negative hidden_dim raises ValueError."""
        config = SwiGLUConfig(hidden_dim=-1, bias=False, gate_dim=4096)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_swiglu_config(config)

    def test_zero_gate_dim_raises_error(self) -> None:
        """Test that zero gate_dim raises ValueError."""
        config = SwiGLUConfig(hidden_dim=4096, bias=False, gate_dim=0)
        with pytest.raises(ValueError, match="gate_dim must be positive"):
            validate_swiglu_config(config)


class TestValidateActivationConfig:
    """Tests for validate_activation_config function."""

    def test_valid_gelu_config(self) -> None:
        """Test validation of valid GELU config."""
        config = create_activation_config(activation_type="gelu")
        validate_activation_config(config)  # Should not raise

    def test_valid_swiglu_config(self) -> None:
        """Test validation of valid SwiGLU config."""
        config = create_activation_config(activation_type="swiglu")
        validate_activation_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_activation_config(None)  # type: ignore[arg-type]

    def test_missing_gelu_config_raises_error(self) -> None:
        """Test that missing gelu_config for GELU type raises error."""
        config = ActivationConfig(
            activation_type=ActivationType.GELU,
            gelu_config=None,
            swiglu_config=None,
            inplace=False,
        )
        with pytest.raises(ValueError, match="gelu_config required"):
            validate_activation_config(config)

    def test_missing_swiglu_config_raises_error(self) -> None:
        """Test that missing swiglu_config for SwiGLU type raises error."""
        config = ActivationConfig(
            activation_type=ActivationType.SWIGLU,
            gelu_config=None,
            swiglu_config=None,
            inplace=False,
        )
        with pytest.raises(ValueError, match="swiglu_config required"):
            validate_activation_config(config)


class TestValidateActivationStats:
    """Tests for validate_activation_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = ActivationStats(
            memory_overhead=1.0,
            compute_cost=1.5,
            gradient_stability=0.9,
        )
        validate_activation_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_activation_stats(None)  # type: ignore[arg-type]

    def test_zero_memory_overhead_raises_error(self) -> None:
        """Test that zero memory_overhead raises ValueError."""
        stats = ActivationStats(
            memory_overhead=0.0,
            compute_cost=1.0,
            gradient_stability=0.5,
        )
        with pytest.raises(ValueError, match="memory_overhead must be positive"):
            validate_activation_stats(stats)

    def test_zero_compute_cost_raises_error(self) -> None:
        """Test that zero compute_cost raises ValueError."""
        stats = ActivationStats(
            memory_overhead=1.0,
            compute_cost=0.0,
            gradient_stability=0.5,
        )
        with pytest.raises(ValueError, match="compute_cost must be positive"):
            validate_activation_stats(stats)

    def test_negative_gradient_stability_raises_error(self) -> None:
        """Test that negative gradient_stability raises ValueError."""
        stats = ActivationStats(
            memory_overhead=1.0,
            compute_cost=1.0,
            gradient_stability=-0.1,
        )
        with pytest.raises(ValueError, match="gradient_stability must be in"):
            validate_activation_stats(stats)

    def test_gradient_stability_above_one_raises_error(self) -> None:
        """Test that gradient_stability > 1 raises ValueError."""
        stats = ActivationStats(
            memory_overhead=1.0,
            compute_cost=1.0,
            gradient_stability=1.1,
        )
        with pytest.raises(ValueError, match="gradient_stability must be in"):
            validate_activation_stats(stats)


class TestCreateGELUConfig:
    """Tests for create_gelu_config function."""

    def test_default_approximation(self) -> None:
        """Test default approximation is none."""
        config = create_gelu_config()
        assert config.approximate == GELUApproximation.NONE

    def test_tanh_approximation(self) -> None:
        """Test tanh approximation."""
        config = create_gelu_config(approximate="tanh")
        assert config.approximate == GELUApproximation.TANH

    def test_sigmoid_approximation(self) -> None:
        """Test sigmoid approximation."""
        config = create_gelu_config(approximate="sigmoid")
        assert config.approximate == GELUApproximation.SIGMOID

    def test_invalid_approximation_raises_error(self) -> None:
        """Test that invalid approximation raises ValueError."""
        with pytest.raises(ValueError, match="approximate must be one of"):
            create_gelu_config(approximate="invalid")


class TestCreateSwiGLUConfig:
    """Tests for create_swiglu_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_swiglu_config()
        assert config.hidden_dim == 4096
        assert config.bias is False
        assert config.gate_dim == 4096

    def test_custom_hidden_dim(self) -> None:
        """Test custom hidden_dim."""
        config = create_swiglu_config(hidden_dim=2048)
        assert config.hidden_dim == 2048
        assert config.gate_dim == 2048  # Defaults to hidden_dim

    def test_custom_gate_dim(self) -> None:
        """Test custom gate_dim."""
        config = create_swiglu_config(hidden_dim=4096, gate_dim=2048)
        assert config.gate_dim == 2048

    def test_with_bias(self) -> None:
        """Test with bias enabled."""
        config = create_swiglu_config(bias=True)
        assert config.bias is True

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            create_swiglu_config(hidden_dim=0)


class TestCreateActivationConfig:
    """Tests for create_activation_config function."""

    def test_default_gelu(self) -> None:
        """Test default creates GELU config."""
        config = create_activation_config()
        assert config.activation_type == ActivationType.GELU
        assert config.gelu_config is not None

    def test_relu(self) -> None:
        """Test ReLU config."""
        config = create_activation_config(activation_type="relu")
        assert config.activation_type == ActivationType.RELU

    def test_swiglu(self) -> None:
        """Test SwiGLU config."""
        config = create_activation_config(activation_type="swiglu")
        assert config.activation_type == ActivationType.SWIGLU
        assert config.swiglu_config is not None

    def test_geglu(self) -> None:
        """Test GeGLU config."""
        config = create_activation_config(activation_type="geglu")
        assert config.activation_type == ActivationType.GEGLU
        assert config.swiglu_config is not None

    def test_inplace(self) -> None:
        """Test inplace option."""
        config = create_activation_config(activation_type="relu", inplace=True)
        assert config.inplace is True

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid activation_type raises ValueError."""
        with pytest.raises(ValueError, match="activation_type must be one of"):
            create_activation_config(activation_type="invalid")

    @given(st.sampled_from(list(VALID_ACTIVATION_TYPES)))
    @settings(max_examples=10)
    def test_all_activation_types_creatable(self, act_type: str) -> None:
        """Test that all valid activation types can be created."""
        config = create_activation_config(activation_type=act_type)
        assert config.activation_type.value == act_type


class TestCreateActivationStats:
    """Tests for create_activation_stats function."""

    def test_relu_stats(self) -> None:
        """Test ReLU stats baseline."""
        stats = create_activation_stats("relu")
        assert stats.compute_cost == pytest.approx(1.0)
        assert stats.memory_overhead == pytest.approx(1.0)
        assert stats.gradient_stability == pytest.approx(0.7)

    def test_gelu_stats(self) -> None:
        """Test GELU stats."""
        stats = create_activation_stats("gelu")
        assert stats.compute_cost > 1.0  # More expensive than ReLU
        assert stats.gradient_stability > 0.7  # Better stability

    def test_swiglu_stats(self) -> None:
        """Test SwiGLU stats."""
        stats = create_activation_stats("swiglu")
        assert stats.memory_overhead == pytest.approx(2.0)  # 2x params
        assert stats.gradient_stability > 0.9  # High stability

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid activation_type raises ValueError."""
        with pytest.raises(ValueError, match="activation_type must be one of"):
            create_activation_stats("invalid")

    @given(st.sampled_from(list(VALID_ACTIVATION_TYPES)))
    @settings(max_examples=10)
    def test_all_types_have_stats(self, act_type: str) -> None:
        """Test that all valid activation types have stats."""
        stats = create_activation_stats(act_type)
        assert stats.memory_overhead > 0
        assert stats.compute_cost > 0
        assert 0 <= stats.gradient_stability <= 1


class TestCalculateActivationMemory:
    """Tests for calculate_activation_memory function."""

    def test_basic_calculation(self) -> None:
        """Test basic memory calculation."""
        mem = calculate_activation_memory(
            batch_size=1,
            seq_length=512,
            hidden_dim=4096,
            activation_type="relu",
        )
        assert mem > 0

    def test_swiglu_more_memory(self) -> None:
        """Test that SwiGLU uses more memory than ReLU."""
        mem_relu = calculate_activation_memory(
            batch_size=1,
            seq_length=512,
            hidden_dim=4096,
            activation_type="relu",
        )
        mem_swiglu = calculate_activation_memory(
            batch_size=1,
            seq_length=512,
            hidden_dim=4096,
            activation_type="swiglu",
        )
        assert mem_swiglu > mem_relu

    def test_fp32_more_memory_than_fp16(self) -> None:
        """Test that fp32 uses more memory than fp16."""
        mem_fp16 = calculate_activation_memory(
            batch_size=1,
            seq_length=512,
            hidden_dim=4096,
            activation_type="relu",
            dtype_bytes=2,
        )
        mem_fp32 = calculate_activation_memory(
            batch_size=1,
            seq_length=512,
            hidden_dim=4096,
            activation_type="relu",
            dtype_bytes=4,
        )
        assert mem_fp32 > mem_fp16

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_activation_memory(
                batch_size=0,
                seq_length=512,
                hidden_dim=4096,
                activation_type="relu",
            )

    def test_zero_seq_length_raises_error(self) -> None:
        """Test that zero seq_length raises ValueError."""
        with pytest.raises(ValueError, match="seq_length must be positive"):
            calculate_activation_memory(
                batch_size=1,
                seq_length=0,
                hidden_dim=4096,
                activation_type="relu",
            )

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            calculate_activation_memory(
                batch_size=1,
                seq_length=512,
                hidden_dim=0,
                activation_type="relu",
            )

    def test_invalid_activation_type_raises_error(self) -> None:
        """Test that invalid activation_type raises ValueError."""
        with pytest.raises(ValueError, match="activation_type must be one of"):
            calculate_activation_memory(
                batch_size=1,
                seq_length=512,
                hidden_dim=4096,
                activation_type="invalid",
            )

    def test_invalid_dtype_bytes_raises_error(self) -> None:
        """Test that invalid dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be"):
            calculate_activation_memory(
                batch_size=1,
                seq_length=512,
                hidden_dim=4096,
                activation_type="relu",
                dtype_bytes=3,
            )


class TestEstimateGLUExpansion:
    """Tests for estimate_glu_expansion function."""

    def test_default_ratio(self) -> None:
        """Test default intermediate ratio."""
        gate, up = estimate_glu_expansion(4096)
        assert gate == up
        assert gate == int(4096 * 4.0)

    def test_custom_ratio(self) -> None:
        """Test custom intermediate ratio."""
        gate, up = estimate_glu_expansion(4096, intermediate_ratio=8.0 / 3.0)
        expected = int(4096 * 8.0 / 3.0)
        assert gate == expected
        assert up == expected

    def test_zero_hidden_dim_raises_error(self) -> None:
        """Test that zero hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            estimate_glu_expansion(0)

    def test_negative_hidden_dim_raises_error(self) -> None:
        """Test that negative hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            estimate_glu_expansion(-1)

    def test_zero_ratio_raises_error(self) -> None:
        """Test that zero intermediate_ratio raises ValueError."""
        with pytest.raises(ValueError, match="intermediate_ratio must be positive"):
            estimate_glu_expansion(4096, intermediate_ratio=0.0)


class TestCompareActivationProperties:
    """Tests for compare_activation_properties function."""

    def test_basic_comparison(self) -> None:
        """Test basic comparison."""
        results = compare_activation_properties(["relu", "gelu", "swiglu"])
        assert "relu" in results
        assert "gelu" in results
        assert "swiglu" in results

    def test_relu_cheaper_than_swiglu(self) -> None:
        """Test that ReLU is cheaper than SwiGLU."""
        results = compare_activation_properties(["relu", "swiglu"])
        assert results["relu"].compute_cost < results["swiglu"].compute_cost

    def test_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compare_activation_properties([])

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="activation_type must be one of"):
            compare_activation_properties(["invalid"])


class TestCalculateGradientMagnitude:
    """Tests for calculate_gradient_magnitude function."""

    def test_relu_positive_input(self) -> None:
        """Test ReLU gradient for positive input."""
        grad = calculate_gradient_magnitude("relu", 1.0)
        assert grad == pytest.approx(1.0)

    def test_relu_negative_input(self) -> None:
        """Test ReLU gradient for negative input."""
        grad = calculate_gradient_magnitude("relu", -1.0)
        assert grad == pytest.approx(0.0)

    def test_sigmoid_at_zero(self) -> None:
        """Test sigmoid gradient at zero."""
        grad = calculate_gradient_magnitude("sigmoid", 0.0)
        assert grad == pytest.approx(0.25, rel=0.01)

    def test_tanh_at_zero(self) -> None:
        """Test tanh gradient at zero."""
        grad = calculate_gradient_magnitude("tanh", 0.0)
        assert grad == pytest.approx(1.0)

    def test_gelu_at_zero(self) -> None:
        """Test GELU gradient at zero."""
        grad = calculate_gradient_magnitude("gelu", 0.0)
        # GELU(0) ~ 0, GELU'(0) ~ 0.5
        assert 0.4 < grad < 0.6

    def test_silu_positive(self) -> None:
        """Test SiLU gradient for positive input."""
        grad = calculate_gradient_magnitude("silu", 1.0)
        assert grad > 0

    def test_mish_gradient(self) -> None:
        """Test Mish gradient."""
        grad = calculate_gradient_magnitude("mish", 0.0)
        assert grad > 0

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid activation_type raises ValueError."""
        with pytest.raises(ValueError, match="activation_type must be one of"):
            calculate_gradient_magnitude("invalid", 0.0)

    @given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    @settings(max_examples=20)
    def test_relu_gradient_is_zero_or_one(self, x: float) -> None:
        """Test that ReLU gradient is always 0 or 1."""
        grad = calculate_gradient_magnitude("relu", x)
        assert grad in (0.0, 1.0)


class TestFormatActivationStats:
    """Tests for format_activation_stats function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        stats = ActivationStats(
            memory_overhead=1.0,
            compute_cost=2.0,
            gradient_stability=0.9,
        )
        formatted = format_activation_stats(stats)
        assert "Memory Overhead: 1.00x" in formatted
        assert "Compute Cost: 2.00x" in formatted
        assert "Gradient Stability: 0.90" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_activation_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedActivationConfig:
    """Tests for get_recommended_activation_config function."""

    def test_llm_default(self) -> None:
        """Test LLM default recommendation."""
        config = get_recommended_activation_config("llm")
        assert config.activation_type == ActivationType.SWIGLU

    def test_llm_efficiency_priority(self) -> None:
        """Test LLM with efficiency priority."""
        config = get_recommended_activation_config("llm", efficiency_priority=True)
        assert config.activation_type == ActivationType.GELU_NEW

    def test_vision_default(self) -> None:
        """Test vision default recommendation."""
        config = get_recommended_activation_config("vision")
        assert config.activation_type == ActivationType.GELU

    def test_encoder_default(self) -> None:
        """Test encoder default recommendation."""
        config = get_recommended_activation_config("encoder")
        assert config.activation_type == ActivationType.GELU

    def test_invalid_model_type_raises_error(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_recommended_activation_config("invalid")


class TestListActivationTypes:
    """Tests for list_activation_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_activation_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_activation_types()
        assert "relu" in types
        assert "gelu" in types
        assert "swiglu" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_activation_types()
        assert types == sorted(types)


class TestListGELUApproximations:
    """Tests for list_gelu_approximations function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_gelu_approximations()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_gelu_approximations()
        assert "none" in methods
        assert "tanh" in methods
        assert "sigmoid" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_gelu_approximations()
        assert methods == sorted(methods)


class TestListGLUVariants:
    """Tests for list_glu_variants function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        variants = list_glu_variants()
        assert isinstance(variants, list)

    def test_contains_expected_variants(self) -> None:
        """Test that list contains expected variants."""
        variants = list_glu_variants()
        assert "swiglu" in variants
        assert "geglu" in variants
        assert "reglu" in variants
        assert "bilinear" in variants

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        variants = list_glu_variants()
        assert variants == sorted(variants)


class TestGetActivationType:
    """Tests for get_activation_type function."""

    def test_valid_types(self) -> None:
        """Test getting valid activation types."""
        assert get_activation_type("relu") == ActivationType.RELU
        assert get_activation_type("gelu") == ActivationType.GELU
        assert get_activation_type("swiglu") == ActivationType.SWIGLU

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid activation type"):
            get_activation_type("invalid")


class TestGetGELUApproximation:
    """Tests for get_gelu_approximation function."""

    def test_valid_approximations(self) -> None:
        """Test getting valid approximations."""
        assert get_gelu_approximation("none") == GELUApproximation.NONE
        assert get_gelu_approximation("tanh") == GELUApproximation.TANH
        assert get_gelu_approximation("sigmoid") == GELUApproximation.SIGMOID

    def test_invalid_approximation_raises_error(self) -> None:
        """Test that invalid approximation raises ValueError."""
        with pytest.raises(ValueError, match="invalid GELU approximation"):
            get_gelu_approximation("invalid")


class TestGetGLUVariant:
    """Tests for get_glu_variant function."""

    def test_valid_variants(self) -> None:
        """Test getting valid GLU variants."""
        assert get_glu_variant("swiglu") == GLUVariant.SWIGLU
        assert get_glu_variant("geglu") == GLUVariant.GEGLU
        assert get_glu_variant("reglu") == GLUVariant.REGLU
        assert get_glu_variant("bilinear") == GLUVariant.BILINEAR

    def test_invalid_variant_raises_error(self) -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="invalid GLU variant"):
            get_glu_variant("invalid")


class TestPropertyBasedTests:
    """Property-based tests using hypothesis."""

    @given(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=10000),
        st.integers(min_value=1, max_value=16384),
        st.sampled_from(list(VALID_ACTIVATION_TYPES)),
    )
    @settings(max_examples=20)
    def test_activation_memory_always_positive(
        self, batch_size: int, seq_length: int, hidden_dim: int, act_type: str
    ) -> None:
        """Test that activation memory is always positive."""
        mem = calculate_activation_memory(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            activation_type=act_type,
        )
        assert mem > 0

    @given(
        st.integers(min_value=64, max_value=16384),
        st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=20)
    def test_glu_expansion_always_positive(
        self, hidden_dim: int, ratio: float
    ) -> None:
        """Test that GLU expansion dimensions are always positive."""
        gate, up = estimate_glu_expansion(hidden_dim, ratio)
        assert gate > 0
        assert up > 0
        assert gate == up
