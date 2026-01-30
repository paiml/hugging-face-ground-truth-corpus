"""Tests for QLoRA fine-tuning utilities."""

from __future__ import annotations

import pytest

from hf_gtc.training.qlora import (
    ComputeType,
    MemoryEstimate,
    QLoRAConfig,
    QLoRATrainingConfig,
    QuantBits,
    QuantConfig,
    QuantType,
    calculate_qlora_trainable_params,
    create_qlora_config,
    create_qlora_training_config,
    create_quant_config,
    estimate_qlora_memory,
    format_memory_estimate,
    get_bnb_config,
    get_compute_type,
    get_qlora_peft_config,
    get_quant_type,
    get_recommended_qlora_config,
    list_compute_types,
    list_quant_bits,
    list_quant_types,
    validate_compute_type,
    validate_qlora_config,
    validate_quant_config,
    validate_quant_type,
)


class TestQuantBits:
    """Tests for QuantBits enum."""

    def test_four_value(self) -> None:
        """Test FOUR value."""
        assert QuantBits.FOUR.value == 4

    def test_eight_value(self) -> None:
        """Test EIGHT value."""
        assert QuantBits.EIGHT.value == 8


class TestComputeType:
    """Tests for ComputeType enum."""

    def test_float16_value(self) -> None:
        """Test FLOAT16 value."""
        assert ComputeType.FLOAT16.value == "float16"

    def test_bfloat16_value(self) -> None:
        """Test BFLOAT16 value."""
        assert ComputeType.BFLOAT16.value == "bfloat16"

    def test_float32_value(self) -> None:
        """Test FLOAT32 value."""
        assert ComputeType.FLOAT32.value == "float32"


class TestQuantType:
    """Tests for QuantType enum."""

    def test_nf4_value(self) -> None:
        """Test NF4 value."""
        assert QuantType.NF4.value == "nf4"

    def test_fp4_value(self) -> None:
        """Test FP4 value."""
        assert QuantType.FP4.value == "fp4"


class TestQuantConfig:
    """Tests for QuantConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = QuantConfig()
        assert config.bits == 4
        assert config.quant_type == QuantType.NF4
        assert config.compute_dtype == ComputeType.FLOAT16
        assert config.double_quant is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = QuantConfig(bits=8, quant_type=QuantType.FP4)
        assert config.bits == 8
        assert config.quant_type == QuantType.FP4

    def test_frozen(self) -> None:
        """Test that QuantConfig is immutable."""
        config = QuantConfig()
        with pytest.raises(AttributeError):
            config.bits = 8  # type: ignore[misc]


class TestQLoRAConfig:
    """Tests for QLoRAConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = QLoRAConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.bias == "none"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = QLoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1

    def test_frozen(self) -> None:
        """Test that QLoRAConfig is immutable."""
        config = QLoRAConfig()
        with pytest.raises(AttributeError):
            config.r = 16  # type: ignore[misc]


class TestQLoRATrainingConfig:
    """Tests for QLoRATrainingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating config."""
        qlora = QLoRAConfig(r=8)
        quant = QuantConfig(bits=4)
        config = QLoRATrainingConfig(qlora_config=qlora, quant_config=quant)
        assert config.qlora_config.r == 8
        assert config.quant_config.bits == 4

    def test_default_values(self) -> None:
        """Test default training values."""
        qlora = QLoRAConfig()
        quant = QuantConfig()
        config = QLoRATrainingConfig(qlora_config=qlora, quant_config=quant)
        assert config.gradient_checkpointing is True
        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 0.3


class TestMemoryEstimate:
    """Tests for MemoryEstimate dataclass."""

    def test_creation(self) -> None:
        """Test creating memory estimate."""
        est = MemoryEstimate(
            model_memory_mb=4000,
            adapter_memory_mb=50,
            optimizer_memory_mb=100,
            activation_memory_mb=500,
            total_memory_mb=4650,
        )
        assert est.model_memory_mb == 4000
        assert est.total_memory_mb == 4650


class TestValidateQuantConfig:
    """Tests for validate_quant_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = QuantConfig(bits=4)
        validate_quant_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_quant_config(None)  # type: ignore[arg-type]

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        config = QuantConfig(bits=3)
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            validate_quant_config(config)

    def test_valid_8bit(self) -> None:
        """Test 8-bit config is valid."""
        config = QuantConfig(bits=8)
        validate_quant_config(config)  # Should not raise


class TestValidateQLoRAConfig:
    """Tests for validate_qlora_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = QLoRAConfig(r=8, lora_alpha=16)
        validate_qlora_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_qlora_config(None)  # type: ignore[arg-type]

    def test_zero_r_raises_error(self) -> None:
        """Test that zero r raises ValueError."""
        config = QLoRAConfig(r=0)
        with pytest.raises(ValueError, match="r must be positive"):
            validate_qlora_config(config)

    def test_zero_alpha_raises_error(self) -> None:
        """Test that zero lora_alpha raises ValueError."""
        config = QLoRAConfig(lora_alpha=0)
        with pytest.raises(ValueError, match="lora_alpha must be positive"):
            validate_qlora_config(config)

    def test_invalid_dropout_raises_error(self) -> None:
        """Test that invalid lora_dropout raises ValueError."""
        config = QLoRAConfig(lora_dropout=1.5)
        with pytest.raises(ValueError, match="lora_dropout must be between"):
            validate_qlora_config(config)

    def test_invalid_bias_raises_error(self) -> None:
        """Test that invalid bias raises ValueError."""
        config = QLoRAConfig(bias="invalid")
        with pytest.raises(ValueError, match="bias must be one of"):
            validate_qlora_config(config)


class TestCreateQuantConfig:
    """Tests for create_quant_config function."""

    def test_creates_config(self) -> None:
        """Test creating quant config."""
        config = create_quant_config(bits=4)
        assert config.bits == 4
        assert config.quant_type == QuantType.NF4

    def test_string_quant_type(self) -> None:
        """Test with string quant type."""
        config = create_quant_config(bits=4, quant_type="fp4")
        assert config.quant_type == QuantType.FP4

    def test_string_compute_dtype(self) -> None:
        """Test with string compute dtype."""
        config = create_quant_config(bits=4, compute_dtype="bfloat16")
        assert config.compute_dtype == ComputeType.BFLOAT16


class TestCreateQLoRAConfig:
    """Tests for create_qlora_config function."""

    def test_creates_config(self) -> None:
        """Test creating qlora config."""
        config = create_qlora_config(r=8, lora_alpha=16)
        assert config.r == 8
        assert config.lora_alpha == 16

    def test_custom_target_modules(self) -> None:
        """Test with custom target modules."""
        config = create_qlora_config(target_modules=["q_proj", "v_proj"])
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_default_target_modules(self) -> None:
        """Test default target modules."""
        config = create_qlora_config()
        assert "q_proj" in config.target_modules


class TestCreateQLoRATrainingConfig:
    """Tests for create_qlora_training_config function."""

    def test_creates_config(self) -> None:
        """Test creating training config."""
        config = create_qlora_training_config()
        assert config.qlora_config.r == 8
        assert config.quant_config.bits == 4

    def test_custom_configs(self) -> None:
        """Test with custom configs."""
        qlora = create_qlora_config(r=16)
        quant = create_quant_config(bits=8)
        config = create_qlora_training_config(
            qlora_config=qlora, quant_config=quant, gradient_accumulation_steps=8
        )
        assert config.qlora_config.r == 16
        assert config.quant_config.bits == 8
        assert config.gradient_accumulation_steps == 8


class TestEstimateQLoRAMemory:
    """Tests for estimate_qlora_memory function."""

    def test_estimates_memory(self) -> None:
        """Test estimating memory."""
        est = estimate_qlora_memory(7_000_000_000, quant_bits=4)
        assert est.model_memory_mb > 0
        assert est.total_memory_mb > est.model_memory_mb

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_qlora_memory(0)

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="quant_bits must be 4 or 8"):
            estimate_qlora_memory(7_000_000_000, quant_bits=3)

    def test_8bit_more_memory(self) -> None:
        """Test that 8-bit uses more memory than 4-bit."""
        est_4bit = estimate_qlora_memory(7_000_000_000, quant_bits=4)
        est_8bit = estimate_qlora_memory(7_000_000_000, quant_bits=8)
        assert est_8bit.model_memory_mb > est_4bit.model_memory_mb


class TestGetQLoRAPeftConfig:
    """Tests for get_qlora_peft_config function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_qlora_config(r=8, lora_alpha=16)
        peft_dict = get_qlora_peft_config(config)
        assert peft_dict["r"] == 8
        assert peft_dict["lora_alpha"] == 16
        assert peft_dict["task_type"] == "CAUSAL_LM"

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_qlora_peft_config(None)  # type: ignore[arg-type]


class TestGetBnbConfig:
    """Tests for get_bnb_config function."""

    def test_4bit_config(self) -> None:
        """Test 4-bit config."""
        config = create_quant_config(bits=4)
        bnb_dict = get_bnb_config(config)
        assert bnb_dict["load_in_4bit"] is True
        assert bnb_dict["bnb_4bit_quant_type"] == "nf4"

    def test_8bit_config(self) -> None:
        """Test 8-bit config."""
        config = create_quant_config(bits=8)
        bnb_dict = get_bnb_config(config)
        assert bnb_dict["load_in_8bit"] is True

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_bnb_config(None)  # type: ignore[arg-type]


class TestCalculateQLoRATrainableParams:
    """Tests for calculate_qlora_trainable_params function."""

    def test_calculates_params(self) -> None:
        """Test calculating params."""
        params, pct = calculate_qlora_trainable_params(7_000_000_000)
        assert params > 0
        assert 0 < pct < 1

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            calculate_qlora_trainable_params(0)

    def test_zero_r_raises_error(self) -> None:
        """Test that zero r raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            calculate_qlora_trainable_params(7_000_000_000, r=0)

    def test_higher_rank_more_params(self) -> None:
        """Test that higher rank means more trainable params."""
        params_8, _ = calculate_qlora_trainable_params(7_000_000_000, r=8)
        params_16, _ = calculate_qlora_trainable_params(7_000_000_000, r=16)
        assert params_16 > params_8


class TestGetRecommendedQLoRAConfig:
    """Tests for get_recommended_qlora_config function."""

    def test_7b_config(self) -> None:
        """Test 7B model config."""
        config = get_recommended_qlora_config("7b")
        assert config.qlora_config.r == 8
        assert config.quant_config.bits == 4

    def test_13b_config(self) -> None:
        """Test 13B model config."""
        config = get_recommended_qlora_config("13b")
        assert config.gradient_accumulation_steps == 8

    def test_70b_config(self) -> None:
        """Test 70B model config."""
        config = get_recommended_qlora_config("70b")
        assert config.qlora_config.r == 16
        assert config.gradient_accumulation_steps == 16

    def test_invalid_size_raises_error(self) -> None:
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="unrecognized model size"):
            get_recommended_qlora_config("invalid")


class TestListQuantTypes:
    """Tests for list_quant_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_quant_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_quant_types()
        assert "nf4" in types
        assert "fp4" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_quant_types()
        assert types == sorted(types)


class TestValidateQuantType:
    """Tests for validate_quant_type function."""

    def test_valid_nf4(self) -> None:
        """Test validation of nf4 type."""
        assert validate_quant_type("nf4") is True

    def test_valid_fp4(self) -> None:
        """Test validation of fp4 type."""
        assert validate_quant_type("fp4") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_quant_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_quant_type("") is False


class TestGetQuantType:
    """Tests for get_quant_type function."""

    def test_get_nf4(self) -> None:
        """Test getting NF4 type."""
        assert get_quant_type("nf4") == QuantType.NF4

    def test_get_fp4(self) -> None:
        """Test getting FP4 type."""
        assert get_quant_type("fp4") == QuantType.FP4

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid quant type"):
            get_quant_type("invalid")


class TestListComputeTypes:
    """Tests for list_compute_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_compute_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_compute_types()
        assert "float16" in types
        assert "bfloat16" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_compute_types()
        assert types == sorted(types)


class TestValidateComputeType:
    """Tests for validate_compute_type function."""

    def test_valid_float16(self) -> None:
        """Test validation of float16 type."""
        assert validate_compute_type("float16") is True

    def test_valid_bfloat16(self) -> None:
        """Test validation of bfloat16 type."""
        assert validate_compute_type("bfloat16") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_compute_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_compute_type("") is False


class TestGetComputeType:
    """Tests for get_compute_type function."""

    def test_get_float16(self) -> None:
        """Test getting FLOAT16 type."""
        assert get_compute_type("float16") == ComputeType.FLOAT16

    def test_get_bfloat16(self) -> None:
        """Test getting BFLOAT16 type."""
        assert get_compute_type("bfloat16") == ComputeType.BFLOAT16

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid compute type"):
            get_compute_type("invalid")


class TestListQuantBits:
    """Tests for list_quant_bits function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        bits = list_quant_bits()
        assert isinstance(bits, list)

    def test_contains_expected_values(self) -> None:
        """Test that list contains expected values."""
        bits = list_quant_bits()
        assert 4 in bits
        assert 8 in bits


class TestFormatMemoryEstimate:
    """Tests for format_memory_estimate function."""

    def test_formats_estimate(self) -> None:
        """Test formatting estimate."""
        est = MemoryEstimate(
            model_memory_mb=4000,
            adapter_memory_mb=50,
            optimizer_memory_mb=100,
            activation_memory_mb=500,
            total_memory_mb=4650,
        )
        formatted = format_memory_estimate(est)
        assert "Model:" in formatted
        assert "Total:" in formatted
        assert "4650" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="estimate cannot be None"):
            format_memory_estimate(None)  # type: ignore[arg-type]
