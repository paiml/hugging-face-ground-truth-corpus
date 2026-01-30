"""Tests for LoRA fine-tuning functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hf_gtc.training.lora import (
    DEFAULT_TARGET_MODULES,
    VALID_BIAS_OPTIONS,
    VALID_TASK_TYPES,
    LoRAConfig,
    TaskType,
    calculate_lora_memory_savings,
    create_lora_config,
    estimate_lora_parameters,
    get_peft_config,
    get_recommended_lora_config,
    list_task_types,
    validate_lora_config,
)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_causal_lm_value(self) -> None:
        """Test CAUSAL_LM enum value."""
        assert TaskType.CAUSAL_LM.value == "CAUSAL_LM"

    def test_seq_cls_value(self) -> None:
        """Test SEQ_CLS enum value."""
        assert TaskType.SEQ_CLS.value == "SEQ_CLS"

    def test_seq_2_seq_lm_value(self) -> None:
        """Test SEQ_2_SEQ_LM enum value."""
        assert TaskType.SEQ_2_SEQ_LM.value == "SEQ_2_SEQ_LM"

    def test_token_cls_value(self) -> None:
        """Test TOKEN_CLS enum value."""
        assert TaskType.TOKEN_CLS.value == "TOKEN_CLS"

    def test_question_ans_value(self) -> None:
        """Test QUESTION_ANS enum value."""
        assert TaskType.QUESTION_ANS.value == "QUESTION_ANS"

    def test_feature_extraction_value(self) -> None:
        """Test FEATURE_EXTRACTION enum value."""
        assert TaskType.FEATURE_EXTRACTION.value == "FEATURE_EXTRACTION"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_TASK_TYPES."""
        for task_type in TaskType:
            assert task_type.value in VALID_TASK_TYPES


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating LoRAConfig instance."""
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=("q_proj", "v_proj"),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            modules_to_save=None,
        )
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == pytest.approx(0.1)
        assert "q_proj" in config.target_modules
        assert config.task_type == TaskType.CAUSAL_LM
        assert config.bias == "none"
        assert config.modules_to_save is None

    def test_frozen(self) -> None:
        """Test that LoRAConfig is immutable."""
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=("q_proj",),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            modules_to_save=None,
        )
        with pytest.raises(AttributeError):
            config.r = 16  # type: ignore[misc]


class TestValidatLoraConfig:
    """Tests for validate_lora_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        validate_lora_config(8, 16, 0.1, "none")  # Should not raise

    def test_zero_r_raises_error(self) -> None:
        """Test that r=0 raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            validate_lora_config(0, 16, 0.1, "none")

    def test_negative_r_raises_error(self) -> None:
        """Test that negative r raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            validate_lora_config(-1, 16, 0.1, "none")

    def test_zero_lora_alpha_raises_error(self) -> None:
        """Test that lora_alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match="lora_alpha must be positive"):
            validate_lora_config(8, 0, 0.1, "none")

    def test_negative_lora_alpha_raises_error(self) -> None:
        """Test that negative lora_alpha raises ValueError."""
        with pytest.raises(ValueError, match="lora_alpha must be positive"):
            validate_lora_config(8, -1, 0.1, "none")

    def test_negative_dropout_raises_error(self) -> None:
        """Test that negative dropout raises ValueError."""
        with pytest.raises(ValueError, match="lora_dropout must be in"):
            validate_lora_config(8, 16, -0.1, "none")

    def test_dropout_one_raises_error(self) -> None:
        """Test that dropout=1.0 raises ValueError."""
        with pytest.raises(ValueError, match="lora_dropout must be in"):
            validate_lora_config(8, 16, 1.0, "none")

    def test_dropout_greater_than_one_raises_error(self) -> None:
        """Test that dropout>1.0 raises ValueError."""
        with pytest.raises(ValueError, match="lora_dropout must be in"):
            validate_lora_config(8, 16, 1.5, "none")

    def test_invalid_bias_raises_error(self) -> None:
        """Test that invalid bias raises ValueError."""
        with pytest.raises(ValueError, match="bias must be one of"):
            validate_lora_config(8, 16, 0.1, "invalid")

    def test_valid_bias_options(self) -> None:
        """Test all valid bias options."""
        for bias in VALID_BIAS_OPTIONS:
            validate_lora_config(8, 16, 0.1, bias)  # Should not raise


class TestCreateLoraConfig:
    """Tests for create_lora_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_lora_config()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == pytest.approx(0.1)
        assert config.task_type == TaskType.CAUSAL_LM
        assert config.bias == "none"

    def test_default_target_modules(self) -> None:
        """Test default target modules are set."""
        config = create_lora_config()
        assert len(config.target_modules) > 0
        for module in DEFAULT_TARGET_MODULES:
            assert module in config.target_modules

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_lora_config(
            r=16,
            lora_alpha=32,
            lora_dropout=0.2,
            target_modules=("q_proj", "v_proj"),
            task_type="SEQ_CLS",
            bias="all",
        )
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == pytest.approx(0.2)
        assert config.target_modules == ("q_proj", "v_proj")
        assert config.task_type == TaskType.SEQ_CLS
        assert config.bias == "all"

    def test_invalid_r_raises_error(self) -> None:
        """Test that invalid r raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            create_lora_config(r=0)

    def test_invalid_task_type_raises_error(self) -> None:
        """Test that invalid task_type raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            create_lora_config(task_type="INVALID")

    def test_modules_to_save(self) -> None:
        """Test modules_to_save parameter."""
        config = create_lora_config(modules_to_save=("lm_head",))
        assert config.modules_to_save == ("lm_head",)


class TestGetPeftConfig:
    """Tests for get_peft_config function."""

    @patch("peft.LoraConfig")
    @patch("peft.TaskType")
    def test_creates_peft_config(
        self, mock_task_type: MagicMock, mock_lora_config: MagicMock
    ) -> None:
        """Test that PEFT config is created correctly."""
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        mock_task_type.SEQ_CLS = "SEQ_CLS"
        mock_task_type.SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
        mock_task_type.TOKEN_CLS = "TOKEN_CLS"
        mock_task_type.QUESTION_ANS = "QUESTION_ANS"
        mock_task_type.FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
        config = create_lora_config()
        get_peft_config(config)
        mock_lora_config.assert_called_once()

    @patch("peft.LoraConfig")
    @patch("peft.TaskType")
    def test_passes_parameters(
        self, mock_task_type: MagicMock, mock_lora_config: MagicMock
    ) -> None:
        """Test that parameters are passed correctly."""
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        mock_task_type.SEQ_CLS = "SEQ_CLS"
        mock_task_type.SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
        mock_task_type.TOKEN_CLS = "TOKEN_CLS"
        mock_task_type.QUESTION_ANS = "QUESTION_ANS"
        mock_task_type.FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
        config = create_lora_config(r=16, lora_alpha=32)
        get_peft_config(config)

        call_kwargs = mock_lora_config.call_args.kwargs
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32


class TestEstimateLoraParameters:
    """Tests for estimate_lora_parameters function."""

    def test_basic_estimation(self) -> None:
        """Test basic parameter estimation."""
        params = estimate_lora_parameters(
            base_model_params=7_000_000_000,
            r=8,
            num_target_modules=7,
            hidden_size=4096,
        )
        # 2 * 4096 * 8 * 7 = 458752
        assert params == 458752

    def test_higher_rank(self) -> None:
        """Test estimation with higher rank."""
        params = estimate_lora_parameters(
            base_model_params=7_000_000_000,
            r=16,
            num_target_modules=7,
            hidden_size=4096,
        )
        # 2 * 4096 * 16 * 7 = 917504
        assert params == 917504

    def test_zero_base_params_raises_error(self) -> None:
        """Test that zero base_model_params raises ValueError."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            estimate_lora_parameters(0, r=8, num_target_modules=7)

    def test_negative_base_params_raises_error(self) -> None:
        """Test that negative base_model_params raises ValueError."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            estimate_lora_parameters(-100, r=8, num_target_modules=7)

    def test_zero_r_raises_error(self) -> None:
        """Test that r=0 raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            estimate_lora_parameters(1000, r=0, num_target_modules=7)

    def test_zero_num_modules_raises_error(self) -> None:
        """Test that num_target_modules=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_target_modules must be positive"):
            estimate_lora_parameters(1000, r=8, num_target_modules=0)

    def test_zero_hidden_size_raises_error(self) -> None:
        """Test that hidden_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_lora_parameters(1000, r=8, num_target_modules=7, hidden_size=0)


class TestCalculateLoraMemorySavings:
    """Tests for calculate_lora_memory_savings function."""

    def test_small_percentage(self) -> None:
        """Test that LoRA gives small percentage of trainable params."""
        savings = calculate_lora_memory_savings(7_000_000_000, 458752)
        assert savings < 0.01  # Less than 0.01%

    def test_larger_lora(self) -> None:
        """Test with larger LoRA configuration."""
        savings = calculate_lora_memory_savings(7_000_000_000, 4_000_000)
        assert savings < 0.1  # Still less than 0.1%

    def test_zero_base_params_raises_error(self) -> None:
        """Test that zero base_model_params raises ValueError."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            calculate_lora_memory_savings(0, 1000)

    def test_negative_base_params_raises_error(self) -> None:
        """Test that negative base_model_params raises ValueError."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            calculate_lora_memory_savings(-100, 1000)

    def test_zero_lora_params_raises_error(self) -> None:
        """Test that zero lora_params raises ValueError."""
        with pytest.raises(ValueError, match="lora_params must be positive"):
            calculate_lora_memory_savings(1000, 0)

    def test_negative_lora_params_raises_error(self) -> None:
        """Test that negative lora_params raises ValueError."""
        with pytest.raises(ValueError, match="lora_params must be positive"):
            calculate_lora_memory_savings(1000, -100)


class TestListTaskTypes:
    """Tests for list_task_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_task_types()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_task_types()
        assert result == sorted(result)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        result = list_task_types()
        assert "CAUSAL_LM" in result
        assert "SEQ_CLS" in result
        assert "TOKEN_CLS" in result

    def test_correct_count(self) -> None:
        """Test correct number of types."""
        result = list_task_types()
        assert len(result) == 6


class TestGetRecommendedLoraConfig:
    """Tests for get_recommended_lora_config function."""

    def test_small_model(self) -> None:
        """Test recommended config for small model."""
        config = get_recommended_lora_config("small")
        assert config.r == 4
        assert config.lora_alpha == 8
        assert config.lora_dropout == pytest.approx(0.05)

    def test_medium_model(self) -> None:
        """Test recommended config for medium model."""
        config = get_recommended_lora_config("medium")
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == pytest.approx(0.1)

    def test_large_model(self) -> None:
        """Test recommended config for large model."""
        config = get_recommended_lora_config("large")
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == pytest.approx(0.1)

    def test_xlarge_model(self) -> None:
        """Test recommended config for xlarge model."""
        config = get_recommended_lora_config("xlarge")
        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == pytest.approx(0.05)

    def test_custom_task_type(self) -> None:
        """Test recommended config with custom task type."""
        config = get_recommended_lora_config("medium", task_type="SEQ_CLS")
        assert config.task_type == TaskType.SEQ_CLS

    def test_invalid_model_size_raises_error(self) -> None:
        """Test that invalid model_size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_lora_config("invalid")
