"""Tests for adapter-based fine-tuning utilities."""

from __future__ import annotations

import pytest

from hf_gtc.training.adapters import (
    ATTENTION_MODULES,
    ALL_MODULES,
    MLP_MODULES,
    VALID_ADAPTER_TYPES,
    VALID_MERGE_STRATEGIES,
    VALID_TARGET_MODULES,
    AdaLoRAConfig,
    AdapterConfig,
    AdapterStats,
    AdapterType,
    IA3Config,
    LoRAConfig,
    MergeStrategy,
    PrefixTuningConfig,
    QLoRAConfig,
    TargetModules,
    calculate_lora_rank,
    calculate_trainable_params,
    create_adalora_config,
    create_adapter_config,
    create_adapter_stats,
    create_ia3_config,
    create_lora_config,
    create_prefix_config,
    create_qlora_config,
    estimate_memory_savings,
    format_adapter_stats,
    get_adapter_type,
    get_merge_strategy,
    get_peft_config_dict,
    get_recommended_adapter_config,
    get_target_modules,
    get_target_modules_list,
    list_adapter_types,
    list_merge_strategies,
    list_target_modules,
    merge_adapter_weights,
    validate_adalora_config,
    validate_adapter_config,
    validate_ia3_config,
    validate_lora_config,
    validate_prefix_config,
    validate_qlora_config,
)


class TestAdapterType:
    """Tests for AdapterType enum."""

    def test_lora_value(self) -> None:
        """Test LORA enum value."""
        assert AdapterType.LORA.value == "lora"

    def test_qlora_value(self) -> None:
        """Test QLORA enum value."""
        assert AdapterType.QLORA.value == "qlora"

    def test_adalora_value(self) -> None:
        """Test ADALORA enum value."""
        assert AdapterType.ADALORA.value == "adalora"

    def test_ia3_value(self) -> None:
        """Test IA3 enum value."""
        assert AdapterType.IA3.value == "ia3"

    def test_prefix_tuning_value(self) -> None:
        """Test PREFIX_TUNING enum value."""
        assert AdapterType.PREFIX_TUNING.value == "prefix_tuning"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_ADAPTER_TYPES."""
        for adapter_type in AdapterType:
            assert adapter_type.value in VALID_ADAPTER_TYPES


class TestTargetModules:
    """Tests for TargetModules enum."""

    def test_attention_value(self) -> None:
        """Test ATTENTION enum value."""
        assert TargetModules.ATTENTION.value == "attention"

    def test_mlp_value(self) -> None:
        """Test MLP enum value."""
        assert TargetModules.MLP.value == "mlp"

    def test_all_value(self) -> None:
        """Test ALL enum value."""
        assert TargetModules.ALL.value == "all"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_TARGET_MODULES."""
        for tm in TargetModules:
            assert tm.value in VALID_TARGET_MODULES


class TestMergeStrategy:
    """Tests for MergeStrategy enum."""

    def test_cat_value(self) -> None:
        """Test CAT enum value."""
        assert MergeStrategy.CAT.value == "cat"

    def test_add_value(self) -> None:
        """Test ADD enum value."""
        assert MergeStrategy.ADD.value == "add"

    def test_weighted_value(self) -> None:
        """Test WEIGHTED enum value."""
        assert MergeStrategy.WEIGHTED.value == "weighted"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_MERGE_STRATEGIES."""
        for ms in MergeStrategy:
            assert ms.value in VALID_MERGE_STRATEGIES


class TestModuleConstants:
    """Tests for module constant frozensets."""

    def test_attention_modules_content(self) -> None:
        """Test ATTENTION_MODULES contains expected modules."""
        assert "q_proj" in ATTENTION_MODULES
        assert "k_proj" in ATTENTION_MODULES
        assert "v_proj" in ATTENTION_MODULES
        assert "o_proj" in ATTENTION_MODULES

    def test_mlp_modules_content(self) -> None:
        """Test MLP_MODULES contains expected modules."""
        assert "gate_proj" in MLP_MODULES
        assert "up_proj" in MLP_MODULES
        assert "down_proj" in MLP_MODULES

    def test_all_modules_is_union(self) -> None:
        """Test ALL_MODULES is union of attention and mlp."""
        assert ALL_MODULES == ATTENTION_MODULES | MLP_MODULES


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating LoRAConfig instance."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=("q_proj", "v_proj"),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        assert config.r == 8
        assert config.alpha == 16
        assert config.dropout == pytest.approx(0.1)
        assert "q_proj" in config.target_modules
        assert config.bias == "none"
        assert config.use_rslora is False
        assert config.use_dora is False

    def test_frozen(self) -> None:
        """Test that LoRAConfig is immutable."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=("q_proj",),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(AttributeError):
            config.r = 16  # type: ignore[misc]


class TestQLoRAConfig:
    """Tests for QLoRAConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating QLoRAConfig instance."""
        config = QLoRAConfig(
            bits=4,
            double_quant=True,
            compute_dtype="float16",
            quant_type="nf4",
        )
        assert config.bits == 4
        assert config.double_quant is True
        assert config.compute_dtype == "float16"
        assert config.quant_type == "nf4"

    def test_frozen(self) -> None:
        """Test that QLoRAConfig is immutable."""
        config = QLoRAConfig(
            bits=4,
            double_quant=True,
            compute_dtype="float16",
            quant_type="nf4",
        )
        with pytest.raises(AttributeError):
            config.bits = 8  # type: ignore[misc]


class TestAdaLoRAConfig:
    """Tests for AdaLoRAConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating AdaLoRAConfig instance."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        assert config.init_r == 12
        assert config.target_r == 8
        assert config.beta1 == pytest.approx(0.85)
        assert config.tinit == 200
        assert config.tfinal == 1000

    def test_frozen(self) -> None:
        """Test that AdaLoRAConfig is immutable."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        with pytest.raises(AttributeError):
            config.init_r = 16  # type: ignore[misc]


class TestIA3Config:
    """Tests for IA3Config dataclass."""

    def test_creation(self) -> None:
        """Test creating IA3Config instance."""
        config = IA3Config(
            target_modules=("q_proj", "v_proj", "down_proj"),
            feedforward_modules=("down_proj",),
            init_weights=True,
        )
        assert "q_proj" in config.target_modules
        assert "down_proj" in config.feedforward_modules
        assert config.init_weights is True

    def test_frozen(self) -> None:
        """Test that IA3Config is immutable."""
        config = IA3Config(
            target_modules=("q_proj",),
            feedforward_modules=(),
            init_weights=True,
        )
        with pytest.raises(AttributeError):
            config.init_weights = False  # type: ignore[misc]


class TestPrefixTuningConfig:
    """Tests for PrefixTuningConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PrefixTuningConfig instance."""
        config = PrefixTuningConfig(
            num_virtual_tokens=20,
            encoder_hidden_size=512,
            prefix_projection=True,
        )
        assert config.num_virtual_tokens == 20
        assert config.encoder_hidden_size == 512
        assert config.prefix_projection is True

    def test_frozen(self) -> None:
        """Test that PrefixTuningConfig is immutable."""
        config = PrefixTuningConfig(
            num_virtual_tokens=20,
            encoder_hidden_size=512,
            prefix_projection=True,
        )
        with pytest.raises(AttributeError):
            config.num_virtual_tokens = 30  # type: ignore[misc]


class TestAdapterStats:
    """Tests for AdapterStats dataclass."""

    def test_creation(self) -> None:
        """Test creating AdapterStats instance."""
        stats = AdapterStats(
            adapter_type=AdapterType.LORA,
            trainable_params=4_000_000,
            total_params=7_000_000_000,
            trainable_percent=0.057,
            memory_saved_mb=12000.0,
        )
        assert stats.adapter_type == AdapterType.LORA
        assert stats.trainable_params == 4_000_000
        assert stats.trainable_percent == pytest.approx(0.057)


class TestValidateLoraConfig:
    """Tests for validate_lora_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=("q_proj",),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        validate_lora_config(config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_lora_config(None)  # type: ignore[arg-type]

    def test_zero_r_raises_error(self) -> None:
        """Test that r=0 raises ValueError."""
        config = LoRAConfig(
            r=0,
            alpha=16,
            dropout=0.1,
            target_modules=("q_proj",),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(ValueError, match="r must be positive"):
            validate_lora_config(config)

    def test_negative_r_raises_error(self) -> None:
        """Test that negative r raises ValueError."""
        config = LoRAConfig(
            r=-1,
            alpha=16,
            dropout=0.1,
            target_modules=("q_proj",),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(ValueError, match="r must be positive"):
            validate_lora_config(config)

    def test_zero_alpha_raises_error(self) -> None:
        """Test that alpha=0 raises ValueError."""
        config = LoRAConfig(
            r=8,
            alpha=0,
            dropout=0.1,
            target_modules=("q_proj",),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(ValueError, match="alpha must be positive"):
            validate_lora_config(config)

    def test_negative_dropout_raises_error(self) -> None:
        """Test that negative dropout raises ValueError."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=-0.1,
            target_modules=("q_proj",),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(ValueError, match="dropout must be in"):
            validate_lora_config(config)

    def test_dropout_one_raises_error(self) -> None:
        """Test that dropout=1.0 raises ValueError."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=1.0,
            target_modules=("q_proj",),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(ValueError, match="dropout must be in"):
            validate_lora_config(config)

    def test_invalid_bias_raises_error(self) -> None:
        """Test that invalid bias raises ValueError."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=("q_proj",),
            bias="invalid",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(ValueError, match="bias must be one of"):
            validate_lora_config(config)

    def test_empty_target_modules_raises_error(self) -> None:
        """Test that empty target_modules raises ValueError."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=(),
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        with pytest.raises(ValueError, match="target_modules cannot be empty"):
            validate_lora_config(config)


class TestValidateQLoRAConfig:
    """Tests for validate_qlora_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = QLoRAConfig(
            bits=4,
            double_quant=True,
            compute_dtype="float16",
            quant_type="nf4",
        )
        validate_qlora_config(config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_qlora_config(None)  # type: ignore[arg-type]

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        config = QLoRAConfig(
            bits=3,
            double_quant=True,
            compute_dtype="float16",
            quant_type="nf4",
        )
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            validate_qlora_config(config)

    def test_invalid_compute_dtype_raises_error(self) -> None:
        """Test that invalid compute_dtype raises ValueError."""
        config = QLoRAConfig(
            bits=4,
            double_quant=True,
            compute_dtype="invalid",
            quant_type="nf4",
        )
        with pytest.raises(ValueError, match="compute_dtype must be one of"):
            validate_qlora_config(config)

    def test_invalid_quant_type_raises_error(self) -> None:
        """Test that invalid quant_type raises ValueError."""
        config = QLoRAConfig(
            bits=4,
            double_quant=True,
            compute_dtype="float16",
            quant_type="invalid",
        )
        with pytest.raises(ValueError, match="quant_type must be one of"):
            validate_qlora_config(config)


class TestValidateAdaLoRAConfig:
    """Tests for validate_adalora_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        validate_adalora_config(config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_adalora_config(None)  # type: ignore[arg-type]

    def test_zero_init_r_raises_error(self) -> None:
        """Test that init_r=0 raises ValueError."""
        config = AdaLoRAConfig(
            init_r=0,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        with pytest.raises(ValueError, match="init_r must be positive"):
            validate_adalora_config(config)

    def test_zero_target_r_raises_error(self) -> None:
        """Test that target_r=0 raises ValueError."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=0,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        with pytest.raises(ValueError, match="target_r must be positive"):
            validate_adalora_config(config)

    def test_target_r_exceeds_init_r_raises_error(self) -> None:
        """Test that target_r > init_r raises ValueError."""
        config = AdaLoRAConfig(
            init_r=8,
            target_r=12,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        with pytest.raises(ValueError, match="target_r .* cannot exceed init_r"):
            validate_adalora_config(config)

    def test_invalid_beta1_raises_error(self) -> None:
        """Test that invalid beta1 raises ValueError."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=1.0,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        with pytest.raises(ValueError, match="beta1 must be in"):
            validate_adalora_config(config)

    def test_invalid_beta2_raises_error(self) -> None:
        """Test that invalid beta2 raises ValueError."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.0,
            tinit=200,
            tfinal=1000,
            delta_t=10,
        )
        with pytest.raises(ValueError, match="beta2 must be in"):
            validate_adalora_config(config)

    def test_negative_tinit_raises_error(self) -> None:
        """Test that negative tinit raises ValueError."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=-1,
            tfinal=1000,
            delta_t=10,
        )
        with pytest.raises(ValueError, match="tinit must be non-negative"):
            validate_adalora_config(config)

    def test_tfinal_not_greater_than_tinit_raises_error(self) -> None:
        """Test that tfinal <= tinit raises ValueError."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=1000,
            tfinal=500,
            delta_t=10,
        )
        with pytest.raises(ValueError, match="tfinal .* must be greater than tinit"):
            validate_adalora_config(config)

    def test_zero_delta_t_raises_error(self) -> None:
        """Test that delta_t=0 raises ValueError."""
        config = AdaLoRAConfig(
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            delta_t=0,
        )
        with pytest.raises(ValueError, match="delta_t must be positive"):
            validate_adalora_config(config)


class TestValidateIA3Config:
    """Tests for validate_ia3_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = IA3Config(
            target_modules=("q_proj", "v_proj"),
            feedforward_modules=("down_proj",),
            init_weights=True,
        )
        validate_ia3_config(config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_ia3_config(None)  # type: ignore[arg-type]

    def test_empty_target_modules_raises_error(self) -> None:
        """Test that empty target_modules raises ValueError."""
        config = IA3Config(
            target_modules=(),
            feedforward_modules=(),
            init_weights=True,
        )
        with pytest.raises(ValueError, match="target_modules cannot be empty"):
            validate_ia3_config(config)


class TestValidatePrefixConfig:
    """Tests for validate_prefix_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = PrefixTuningConfig(
            num_virtual_tokens=20,
            encoder_hidden_size=512,
            prefix_projection=True,
        )
        validate_prefix_config(config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_prefix_config(None)  # type: ignore[arg-type]

    def test_zero_virtual_tokens_raises_error(self) -> None:
        """Test that num_virtual_tokens=0 raises ValueError."""
        config = PrefixTuningConfig(
            num_virtual_tokens=0,
            encoder_hidden_size=512,
            prefix_projection=True,
        )
        with pytest.raises(ValueError, match="num_virtual_tokens must be positive"):
            validate_prefix_config(config)

    def test_zero_hidden_size_raises_error(self) -> None:
        """Test that encoder_hidden_size=0 raises ValueError."""
        config = PrefixTuningConfig(
            num_virtual_tokens=20,
            encoder_hidden_size=0,
            prefix_projection=True,
        )
        with pytest.raises(ValueError, match="encoder_hidden_size must be positive"):
            validate_prefix_config(config)


class TestValidateAdapterConfig:
    """Tests for validate_adapter_config function."""

    def test_valid_lora_config(self) -> None:
        """Test validation passes for valid LoRA config."""
        config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        validate_adapter_config(config)

    def test_valid_qlora_config(self) -> None:
        """Test validation passes for valid QLoRA config."""
        config = create_adapter_config(adapter_type="qlora", r=8, alpha=16, bits=4)
        validate_adapter_config(config)

    def test_valid_adalora_config(self) -> None:
        """Test validation passes for valid AdaLoRA config."""
        config = create_adapter_config(adapter_type="adalora", r=8, alpha=16)
        validate_adapter_config(config)

    def test_valid_ia3_config(self) -> None:
        """Test validation passes for valid IA3 config."""
        config = create_adapter_config(adapter_type="ia3")
        validate_adapter_config(config)

    def test_valid_prefix_config(self) -> None:
        """Test validation passes for valid prefix tuning config."""
        config = create_adapter_config(adapter_type="prefix_tuning")
        validate_adapter_config(config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_adapter_config(None)  # type: ignore[arg-type]


class TestValidateAdapterConfigMissingConfigs:
    """Tests for validate_adapter_config with missing sub-configs."""

    def test_missing_lora_config_for_lora(self) -> None:
        """Test that missing lora_config for lora raises ValueError."""
        config = AdapterConfig(
            adapter_type=AdapterType.LORA,
            lora_config=None,
            qlora_config=None,
            adalora_config=None,
            ia3_config=None,
            prefix_config=None,
            trainable_params=1000,
            base_model_params=7_000_000_000,
        )
        with pytest.raises(ValueError, match="lora_config required"):
            validate_adapter_config(config)

    def test_missing_qlora_config_for_qlora(self) -> None:
        """Test that missing qlora_config for qlora raises ValueError."""
        lora = create_lora_config()
        config = AdapterConfig(
            adapter_type=AdapterType.QLORA,
            lora_config=lora,
            qlora_config=None,
            adalora_config=None,
            ia3_config=None,
            prefix_config=None,
            trainable_params=1000,
            base_model_params=7_000_000_000,
        )
        with pytest.raises(ValueError, match="qlora_config required"):
            validate_adapter_config(config)

    def test_missing_adalora_config_for_adalora(self) -> None:
        """Test that missing adalora_config for adalora raises ValueError."""
        lora = create_lora_config()
        config = AdapterConfig(
            adapter_type=AdapterType.ADALORA,
            lora_config=lora,
            qlora_config=None,
            adalora_config=None,
            ia3_config=None,
            prefix_config=None,
            trainable_params=1000,
            base_model_params=7_000_000_000,
        )
        with pytest.raises(ValueError, match="adalora_config required"):
            validate_adapter_config(config)

    def test_missing_ia3_config_for_ia3(self) -> None:
        """Test that missing ia3_config for ia3 raises ValueError."""
        config = AdapterConfig(
            adapter_type=AdapterType.IA3,
            lora_config=None,
            qlora_config=None,
            adalora_config=None,
            ia3_config=None,
            prefix_config=None,
            trainable_params=1000,
            base_model_params=7_000_000_000,
        )
        with pytest.raises(ValueError, match="ia3_config required"):
            validate_adapter_config(config)

    def test_missing_prefix_config_for_prefix_tuning(self) -> None:
        """Test that missing prefix_config for prefix_tuning raises ValueError."""
        config = AdapterConfig(
            adapter_type=AdapterType.PREFIX_TUNING,
            lora_config=None,
            qlora_config=None,
            adalora_config=None,
            ia3_config=None,
            prefix_config=None,
            trainable_params=1000,
            base_model_params=7_000_000_000,
        )
        with pytest.raises(ValueError, match="prefix_config required"):
            validate_adapter_config(config)

    def test_invalid_base_model_params(self) -> None:
        """Test that invalid base_model_params raises ValueError."""
        lora = create_lora_config()
        config = AdapterConfig(
            adapter_type=AdapterType.LORA,
            lora_config=lora,
            qlora_config=None,
            adalora_config=None,
            ia3_config=None,
            prefix_config=None,
            trainable_params=1000,
            base_model_params=0,
        )
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            validate_adapter_config(config)


class TestCreateLoRAConfig:
    """Tests for create_lora_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_lora_config()
        assert config.r == 8
        assert config.alpha == 16
        assert config.dropout == pytest.approx(0.1)
        assert config.bias == "none"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_lora_config(r=16, alpha=32, dropout=0.2)
        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == pytest.approx(0.2)

    def test_target_modules_attention(self) -> None:
        """Test attention preset for target_modules."""
        config = create_lora_config(target_modules="attention")
        assert "q_proj" in config.target_modules
        assert "gate_proj" not in config.target_modules

    def test_target_modules_mlp(self) -> None:
        """Test mlp preset for target_modules."""
        config = create_lora_config(target_modules="mlp")
        assert "down_proj" in config.target_modules
        assert "q_proj" not in config.target_modules

    def test_target_modules_all(self) -> None:
        """Test all preset for target_modules."""
        config = create_lora_config(target_modules="all")
        assert "q_proj" in config.target_modules
        assert "down_proj" in config.target_modules

    def test_custom_target_modules(self) -> None:
        """Test custom target_modules tuple."""
        config = create_lora_config(target_modules=("q_proj", "v_proj"))
        assert config.target_modules == ("q_proj", "v_proj")

    def test_invalid_preset_raises_error(self) -> None:
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="target_modules preset must be"):
            create_lora_config(target_modules="invalid")

    def test_invalid_r_raises_error(self) -> None:
        """Test that invalid r raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            create_lora_config(r=0)

    def test_use_rslora(self) -> None:
        """Test use_rslora parameter."""
        config = create_lora_config(use_rslora=True)
        assert config.use_rslora is True

    def test_use_dora(self) -> None:
        """Test use_dora parameter."""
        config = create_lora_config(use_dora=True)
        assert config.use_dora is True


class TestCreateQLoRAConfig:
    """Tests for create_qlora_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_qlora_config()
        assert config.bits == 4
        assert config.double_quant is True
        assert config.compute_dtype == "float16"
        assert config.quant_type == "nf4"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_qlora_config(bits=8, quant_type="fp4")
        assert config.bits == 8
        assert config.quant_type == "fp4"

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            create_qlora_config(bits=3)


class TestCreateAdaLoRAConfig:
    """Tests for create_adalora_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_adalora_config()
        assert config.init_r == 12
        assert config.target_r == 8
        assert config.beta1 == pytest.approx(0.85)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_adalora_config(init_r=16, target_r=8)
        assert config.init_r == 16
        assert config.target_r == 8

    def test_invalid_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        with pytest.raises(ValueError, match="init_r must be positive"):
            create_adalora_config(init_r=0)


class TestCreateIA3Config:
    """Tests for create_ia3_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_ia3_config()
        assert "q_proj" in config.target_modules
        assert config.init_weights is True

    def test_custom_target_modules(self) -> None:
        """Test custom target_modules."""
        config = create_ia3_config(target_modules=("k_proj", "v_proj", "down_proj"))
        assert "k_proj" in config.target_modules

    def test_auto_feedforward_modules(self) -> None:
        """Test auto-detection of feedforward modules."""
        config = create_ia3_config(target_modules=("q_proj", "down_proj"))
        assert "down_proj" in config.feedforward_modules
        assert "q_proj" not in config.feedforward_modules

    def test_empty_raises_error(self) -> None:
        """Test that empty target_modules raises ValueError."""
        with pytest.raises(ValueError, match="target_modules cannot be empty"):
            create_ia3_config(target_modules=())

    def test_attention_preset(self) -> None:
        """Test attention preset for ia3."""
        config = create_ia3_config(target_modules="attention")
        assert "q_proj" in config.target_modules
        assert "gate_proj" not in config.target_modules

    def test_mlp_preset(self) -> None:
        """Test mlp preset for ia3."""
        config = create_ia3_config(target_modules="mlp")
        assert "down_proj" in config.target_modules
        assert "q_proj" not in config.target_modules

    def test_invalid_preset_raises_error(self) -> None:
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="target_modules preset must be"):
            create_ia3_config(target_modules="invalid")


class TestCreatePrefixConfig:
    """Tests for create_prefix_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_prefix_config()
        assert config.num_virtual_tokens == 20
        assert config.encoder_hidden_size == 512
        assert config.prefix_projection is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_prefix_config(num_virtual_tokens=30)
        assert config.num_virtual_tokens == 30

    def test_invalid_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        with pytest.raises(ValueError, match="num_virtual_tokens must be positive"):
            create_prefix_config(num_virtual_tokens=0)


class TestCreateAdapterConfig:
    """Tests for create_adapter_config function."""

    def test_lora_config(self) -> None:
        """Test creating LoRA adapter config."""
        config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        assert config.adapter_type == AdapterType.LORA
        assert config.lora_config is not None
        assert config.lora_config.r == 8
        assert config.qlora_config is None

    def test_qlora_config(self) -> None:
        """Test creating QLoRA adapter config."""
        config = create_adapter_config(adapter_type="qlora", bits=4)
        assert config.adapter_type == AdapterType.QLORA
        assert config.lora_config is not None
        assert config.qlora_config is not None
        assert config.qlora_config.bits == 4

    def test_adalora_config(self) -> None:
        """Test creating AdaLoRA adapter config."""
        config = create_adapter_config(adapter_type="adalora", init_r=12, target_r=8)
        assert config.adapter_type == AdapterType.ADALORA
        assert config.lora_config is not None
        assert config.adalora_config is not None
        assert config.adalora_config.init_r == 12

    def test_ia3_config(self) -> None:
        """Test creating IA3 adapter config."""
        config = create_adapter_config(adapter_type="ia3")
        assert config.adapter_type == AdapterType.IA3
        assert config.ia3_config is not None
        assert config.lora_config is None

    def test_prefix_tuning_config(self) -> None:
        """Test creating prefix tuning adapter config."""
        config = create_adapter_config(adapter_type="prefix_tuning", num_virtual_tokens=30)
        assert config.adapter_type == AdapterType.PREFIX_TUNING
        assert config.prefix_config is not None
        assert config.prefix_config.num_virtual_tokens == 30

    def test_invalid_adapter_type_raises_error(self) -> None:
        """Test that invalid adapter_type raises ValueError."""
        with pytest.raises(ValueError, match="adapter_type must be one of"):
            create_adapter_config(adapter_type="invalid")

    def test_trainable_params_calculated(self) -> None:
        """Test that trainable_params is calculated."""
        config = create_adapter_config(adapter_type="lora", r=8)
        assert config.trainable_params > 0
        assert config.trainable_params < config.base_model_params


class TestCalculateTrainableParams:
    """Tests for calculate_trainable_params function."""

    def test_lora_params(self) -> None:
        """Test calculating LoRA trainable params."""
        params = calculate_trainable_params("lora", 7_000_000_000, r=8)
        assert params > 0
        assert params < 7_000_000_000

    def test_ia3_fewer_params(self) -> None:
        """Test that IA3 has fewer params than LoRA."""
        lora_params = calculate_trainable_params("lora", 7_000_000_000, r=8)
        ia3_params = calculate_trainable_params("ia3", 7_000_000_000)
        assert ia3_params < lora_params

    def test_adalora_params(self) -> None:
        """Test calculating AdaLoRA params."""
        lora_params = calculate_trainable_params("lora", 7_000_000_000, r=8)
        adalora_params = calculate_trainable_params("adalora", 7_000_000_000, r=8)
        # AdaLoRA has slightly more params due to importance scores
        assert adalora_params > lora_params

    def test_prefix_tuning_params(self) -> None:
        """Test calculating prefix tuning params."""
        params = calculate_trainable_params(
            "prefix_tuning", 7_000_000_000, num_virtual_tokens=20
        )
        assert params > 0

    def test_zero_base_params_raises_error(self) -> None:
        """Test that zero base_model_params raises ValueError."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            calculate_trainable_params("lora", 0)

    def test_invalid_adapter_type_raises_error(self) -> None:
        """Test that invalid adapter_type raises ValueError."""
        with pytest.raises(ValueError, match="adapter_type must be one of"):
            calculate_trainable_params("invalid", 7_000_000_000)


class TestEstimateMemorySavings:
    """Tests for estimate_memory_savings function."""

    def test_lora_savings(self) -> None:
        """Test estimating LoRA memory savings."""
        saved_mb, pct = estimate_memory_savings("lora", 7_000_000_000, 4_000_000)
        assert saved_mb > 0
        assert pct > 80  # At least 80% savings

    def test_qlora_savings(self) -> None:
        """Test estimating QLoRA memory savings."""
        saved_mb, pct = estimate_memory_savings("qlora", 7_000_000_000, 4_000_000, bits=4)
        assert pct > 70

    def test_zero_base_params_raises_error(self) -> None:
        """Test that zero base_model_params raises ValueError."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            estimate_memory_savings("lora", 0, 1000)

    def test_zero_trainable_params_raises_error(self) -> None:
        """Test that zero trainable_params raises ValueError."""
        with pytest.raises(ValueError, match="trainable_params must be positive"):
            estimate_memory_savings("lora", 7_000_000_000, 0)

    def test_invalid_adapter_type_raises_error(self) -> None:
        """Test that invalid adapter_type raises ValueError."""
        with pytest.raises(ValueError, match="adapter_type must be one of"):
            estimate_memory_savings("invalid", 7_000_000_000, 4_000_000)

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="bits must be 4, 8, 16, or 32"):
            estimate_memory_savings("lora", 7_000_000_000, 4_000_000, bits=5)


class TestMergeAdapterWeights:
    """Tests for merge_adapter_weights function."""

    def test_add_strategy(self) -> None:
        """Test add merge strategy."""
        result = merge_adapter_weights([1.0, 2.0], [0.1, 0.2], strategy="add")
        assert result == pytest.approx([1.1, 2.2])

    def test_cat_strategy(self) -> None:
        """Test cat merge strategy."""
        result = merge_adapter_weights([1.0], [2.0], strategy="cat")
        assert result == [1.0, 2.0]

    def test_weighted_strategy(self) -> None:
        """Test weighted merge strategy."""
        result = merge_adapter_weights([1.0, 2.0], [0.1, 0.2], strategy="weighted", alpha=0.5)
        assert result == pytest.approx([1.05, 2.1])

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            merge_adapter_weights([1.0], [2.0], strategy="invalid")

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError for add."""
        with pytest.raises(ValueError, match="weights must have same length"):
            merge_adapter_weights([1.0], [2.0, 3.0], strategy="add")

    def test_invalid_alpha_raises_error(self) -> None:
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            merge_adapter_weights([1.0], [2.0], strategy="weighted", alpha=1.5)

    def test_zero_alpha_raises_error(self) -> None:
        """Test that alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            merge_adapter_weights([1.0], [2.0], strategy="weighted", alpha=0.0)


class TestCalculateLoraRank:
    """Tests for calculate_lora_rank function."""

    def test_basic_calculation(self) -> None:
        """Test basic rank calculation."""
        rank = calculate_lora_rank(7_000_000_000, memory_budget_mb=100)
        assert 1 <= rank <= 64

    def test_larger_budget_larger_rank(self) -> None:
        """Test that larger budget gives larger rank."""
        rank_small = calculate_lora_rank(7_000_000_000, memory_budget_mb=50)
        rank_large = calculate_lora_rank(7_000_000_000, memory_budget_mb=200)
        assert rank_small <= rank_large

    def test_zero_base_params_raises_error(self) -> None:
        """Test that zero base_model_params raises ValueError."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            calculate_lora_rank(0, 100)

    def test_zero_memory_budget_raises_error(self) -> None:
        """Test that zero memory_budget raises ValueError."""
        with pytest.raises(ValueError, match="memory_budget_mb must be positive"):
            calculate_lora_rank(7_000_000_000, 0)


class TestFormatAdapterStats:
    """Tests for format_adapter_stats function."""

    def test_formats_stats(self) -> None:
        """Test formatting adapter stats."""
        stats = AdapterStats(
            adapter_type=AdapterType.LORA,
            trainable_params=4_000_000,
            total_params=7_000_000_000,
            trainable_percent=0.057,
            memory_saved_mb=12000.0,
        )
        formatted = format_adapter_stats(stats)
        assert "Adapter Type: lora" in formatted
        assert "Trainable:" in formatted
        assert "Memory Saved:" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_adapter_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedAdapterConfig:
    """Tests for get_recommended_adapter_config function."""

    def test_7b_config(self) -> None:
        """Test recommended config for 7B model."""
        config = get_recommended_adapter_config("7b")
        assert config.adapter_type == AdapterType.LORA
        assert config.lora_config is not None
        assert config.lora_config.r == 8

    def test_13b_config(self) -> None:
        """Test recommended config for 13B model."""
        config = get_recommended_adapter_config("13b")
        assert config.adapter_type == AdapterType.LORA
        assert config.lora_config is not None
        assert config.lora_config.r == 8

    def test_70b_config(self) -> None:
        """Test recommended config for 70B model."""
        config = get_recommended_adapter_config("70b")
        assert config.adapter_type == AdapterType.QLORA
        assert config.lora_config is not None
        assert config.lora_config.r == 16

    def test_memory_constrained(self) -> None:
        """Test config with memory constraint."""
        config = get_recommended_adapter_config("7b", memory_constraint_gb=8)
        # Should use QLoRA due to tight memory
        assert config.adapter_type == AdapterType.QLORA

    def test_invalid_size_raises_error(self) -> None:
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_adapter_config("invalid")


class TestListFunctions:
    """Tests for list functions."""

    def test_list_adapter_types(self) -> None:
        """Test listing adapter types."""
        types = list_adapter_types()
        assert isinstance(types, list)
        assert "lora" in types
        assert "qlora" in types
        assert types == sorted(types)

    def test_list_target_modules(self) -> None:
        """Test listing target modules."""
        modules = list_target_modules()
        assert isinstance(modules, list)
        assert "attention" in modules
        assert "mlp" in modules
        assert modules == sorted(modules)

    def test_list_merge_strategies(self) -> None:
        """Test listing merge strategies."""
        strategies = list_merge_strategies()
        assert isinstance(strategies, list)
        assert "add" in strategies
        assert "cat" in strategies
        assert strategies == sorted(strategies)


class TestGetFunctions:
    """Tests for get functions."""

    def test_get_adapter_type(self) -> None:
        """Test getting adapter type."""
        assert get_adapter_type("lora") == AdapterType.LORA
        assert get_adapter_type("qlora") == AdapterType.QLORA

    def test_get_adapter_type_invalid(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid adapter type"):
            get_adapter_type("invalid")

    def test_get_target_modules(self) -> None:
        """Test getting target modules."""
        assert get_target_modules("attention") == TargetModules.ATTENTION
        assert get_target_modules("all") == TargetModules.ALL

    def test_get_target_modules_invalid(self) -> None:
        """Test that invalid modules raises ValueError."""
        with pytest.raises(ValueError, match="invalid target modules"):
            get_target_modules("invalid")

    def test_get_merge_strategy(self) -> None:
        """Test getting merge strategy."""
        assert get_merge_strategy("add") == MergeStrategy.ADD
        assert get_merge_strategy("cat") == MergeStrategy.CAT

    def test_get_merge_strategy_invalid(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid merge strategy"):
            get_merge_strategy("invalid")

    def test_get_target_modules_list_attention(self) -> None:
        """Test getting attention modules list."""
        modules = get_target_modules_list("attention")
        assert "q_proj" in modules
        assert "gate_proj" not in modules

    def test_get_target_modules_list_mlp(self) -> None:
        """Test getting mlp modules list."""
        modules = get_target_modules_list("mlp")
        assert "down_proj" in modules
        assert "q_proj" not in modules

    def test_get_target_modules_list_all(self) -> None:
        """Test getting all modules list."""
        modules = get_target_modules_list("all")
        assert "down_proj" in modules
        assert "q_proj" in modules

    def test_get_target_modules_list_invalid(self) -> None:
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="preset must be one of"):
            get_target_modules_list("invalid")


class TestCreateAdapterStats:
    """Tests for create_adapter_stats function."""

    def test_creates_stats(self) -> None:
        """Test creating adapter stats."""
        config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        stats = create_adapter_stats(config)
        assert stats.adapter_type == AdapterType.LORA
        assert stats.trainable_params > 0
        assert stats.trainable_percent < 1.0

    def test_creates_stats_qlora(self) -> None:
        """Test creating adapter stats for QLoRA."""
        config = create_adapter_config(adapter_type="qlora", r=8, alpha=16, bits=4)
        stats = create_adapter_stats(config)
        assert stats.adapter_type == AdapterType.QLORA
        assert stats.memory_saved_mb > 0

    def test_none_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="adapter_config cannot be None"):
            create_adapter_stats(None)  # type: ignore[arg-type]


class TestGetPeftConfigDict:
    """Tests for get_peft_config_dict function."""

    def test_lora_config(self) -> None:
        """Test getting PEFT dict for LoRA."""
        config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        peft_dict = get_peft_config_dict(config)
        assert peft_dict["r"] == 8
        assert peft_dict["lora_alpha"] == 16
        assert peft_dict["peft_type"] == "LORA"

    def test_adalora_config(self) -> None:
        """Test getting PEFT dict for AdaLoRA."""
        config = create_adapter_config(adapter_type="adalora", init_r=12, target_r=8)
        peft_dict = get_peft_config_dict(config)
        assert peft_dict["init_r"] == 12
        assert peft_dict["target_r"] == 8

    def test_ia3_config(self) -> None:
        """Test getting PEFT dict for IA3."""
        config = create_adapter_config(adapter_type="ia3")
        peft_dict = get_peft_config_dict(config)
        assert "target_modules" in peft_dict
        assert "feedforward_modules" in peft_dict

    def test_prefix_config(self) -> None:
        """Test getting PEFT dict for prefix tuning."""
        config = create_adapter_config(adapter_type="prefix_tuning", num_virtual_tokens=30)
        peft_dict = get_peft_config_dict(config)
        assert peft_dict["num_virtual_tokens"] == 30

    def test_none_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="adapter_config cannot be None"):
            get_peft_config_dict(None)  # type: ignore[arg-type]
