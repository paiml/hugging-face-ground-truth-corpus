"""Tests for TRL training functionality."""

from __future__ import annotations

import pytest

from hf_gtc.training.trl import (
    VALID_DPO_LOSS_TYPES,
    VALID_TRAINING_METHODS,
    DPOConfig,
    PPOConfig,
    SFTConfig,
    TrainingMethod,
    calculate_kl_penalty,
    create_dpo_config,
    create_ppo_config,
    create_sft_config,
    estimate_reward_model_params,
    get_default_reward_prompt,
    get_recommended_beta,
    list_training_methods,
    validate_dpo_config,
    validate_ppo_config,
    validate_sft_config,
)


class TestTrainingMethod:
    """Tests for TrainingMethod enum."""

    def test_dpo_value(self) -> None:
        """Test DPO method value."""
        assert TrainingMethod.DPO.value == "dpo"

    def test_ppo_value(self) -> None:
        """Test PPO method value."""
        assert TrainingMethod.PPO.value == "ppo"

    def test_sft_value(self) -> None:
        """Test SFT method value."""
        assert TrainingMethod.SFT.value == "sft"

    def test_orpo_value(self) -> None:
        """Test ORPO method value."""
        assert TrainingMethod.ORPO.value == "orpo"

    def test_kto_value(self) -> None:
        """Test KTO method value."""
        assert TrainingMethod.KTO.value == "kto"

    def test_all_methods_in_valid_set(self) -> None:
        """Test all enum values are in VALID_TRAINING_METHODS."""
        for method in TrainingMethod:
            assert method.value in VALID_TRAINING_METHODS


class TestDPOConfig:
    """Tests for DPOConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        config = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            label_smoothing=0.0,
            generate_during_eval=False,
        )
        assert config.beta == pytest.approx(0.1)
        assert config.loss_type == "sigmoid"
        assert config.max_length == 512
        assert config.max_prompt_length == 256
        assert config.label_smoothing == pytest.approx(0.0)
        assert config.generate_during_eval is False

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            label_smoothing=0.0,
            generate_during_eval=False,
        )
        with pytest.raises(AttributeError):
            config.beta = 0.2  # type: ignore[misc]


class TestPPOConfig:
    """Tests for PPOConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            gamma=1.0,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            max_grad_norm=1.0,
        )
        assert config.learning_rate == pytest.approx(1e-5)
        assert config.batch_size == 128
        assert config.ppo_epochs == 4

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            gamma=1.0,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            max_grad_norm=1.0,
        )
        with pytest.raises(AttributeError):
            config.learning_rate = 1e-4  # type: ignore[misc]


class TestSFTConfig:
    """Tests for SFTConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        config = SFTConfig(
            max_seq_length=2048,
            packing=True,
            dataset_text_field="text",
            num_train_epochs=3,
            learning_rate=2e-5,
        )
        assert config.max_seq_length == 2048
        assert config.packing is True
        assert config.dataset_text_field == "text"

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = SFTConfig(
            max_seq_length=2048,
            packing=True,
            dataset_text_field="text",
            num_train_epochs=3,
            learning_rate=2e-5,
        )
        with pytest.raises(AttributeError):
            config.packing = False  # type: ignore[misc]


class TestValidateDPOConfig:
    """Tests for validate_dpo_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            label_smoothing=0.0,
            generate_during_eval=False,
        )
        validate_dpo_config(config)  # Should not raise

    def test_zero_beta_raises(self) -> None:
        """Test that zero beta raises error."""
        config = DPOConfig(
            beta=0.0,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            label_smoothing=0.0,
            generate_during_eval=False,
        )
        with pytest.raises(ValueError, match="beta must be positive"):
            validate_dpo_config(config)

    def test_negative_beta_raises(self) -> None:
        """Test that negative beta raises error."""
        config = DPOConfig(
            beta=-0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            label_smoothing=0.0,
            generate_during_eval=False,
        )
        with pytest.raises(ValueError, match="beta must be positive"):
            validate_dpo_config(config)

    def test_zero_max_length_raises(self) -> None:
        """Test that zero max_length raises error."""
        config = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            max_length=0,
            max_prompt_length=256,
            label_smoothing=0.0,
            generate_during_eval=False,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_dpo_config(config)

    def test_prompt_length_exceeds_max_raises(self) -> None:
        """Test that max_prompt_length >= max_length raises error."""
        config = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=512,
            label_smoothing=0.0,
            generate_during_eval=False,
        )
        with pytest.raises(ValueError, match=r"max_prompt_length.*must be less than"):
            validate_dpo_config(config)

    def test_invalid_label_smoothing_raises(self) -> None:
        """Test that invalid label_smoothing raises error."""
        config = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            label_smoothing=1.0,
            generate_during_eval=False,
        )
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            validate_dpo_config(config)


class TestValidatePPOConfig:
    """Tests for validate_ppo_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            gamma=1.0,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            max_grad_norm=1.0,
        )
        validate_ppo_config(config)  # Should not raise

    def test_zero_learning_rate_raises(self) -> None:
        """Test that zero learning_rate raises error."""
        config = PPOConfig(
            learning_rate=0.0,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            gamma=1.0,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            max_grad_norm=1.0,
        )
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_ppo_config(config)

    def test_mini_batch_exceeds_batch_raises(self) -> None:
        """Test that mini_batch_size > batch_size raises error."""
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=32,
            mini_batch_size=64,
            ppo_epochs=4,
            gamma=1.0,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            max_grad_norm=1.0,
        )
        with pytest.raises(ValueError, match=r"mini_batch_size.*cannot exceed"):
            validate_ppo_config(config)

    def test_gamma_out_of_range_raises(self) -> None:
        """Test that gamma > 1.0 raises error."""
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            gamma=1.5,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            max_grad_norm=1.0,
        )
        with pytest.raises(ValueError, match="gamma must be in"):
            validate_ppo_config(config)

    def test_zero_cliprange_raises(self) -> None:
        """Test that zero cliprange raises error."""
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            gamma=1.0,
            lam=0.95,
            cliprange=0.0,
            cliprange_value=0.2,
            vf_coef=0.1,
            max_grad_norm=1.0,
        )
        with pytest.raises(ValueError, match="cliprange must be positive"):
            validate_ppo_config(config)


class TestValidateSFTConfig:
    """Tests for validate_sft_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = SFTConfig(
            max_seq_length=2048,
            packing=True,
            dataset_text_field="text",
            num_train_epochs=3,
            learning_rate=2e-5,
        )
        validate_sft_config(config)  # Should not raise

    def test_zero_max_seq_length_raises(self) -> None:
        """Test that zero max_seq_length raises error."""
        config = SFTConfig(
            max_seq_length=0,
            packing=True,
            dataset_text_field="text",
            num_train_epochs=3,
            learning_rate=2e-5,
        )
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            validate_sft_config(config)

    def test_empty_text_field_raises(self) -> None:
        """Test that empty dataset_text_field raises error."""
        config = SFTConfig(
            max_seq_length=2048,
            packing=True,
            dataset_text_field="",
            num_train_epochs=3,
            learning_rate=2e-5,
        )
        with pytest.raises(ValueError, match="dataset_text_field cannot be empty"):
            validate_sft_config(config)

    def test_zero_epochs_raises(self) -> None:
        """Test that zero num_train_epochs raises error."""
        config = SFTConfig(
            max_seq_length=2048,
            packing=True,
            dataset_text_field="text",
            num_train_epochs=0,
            learning_rate=2e-5,
        )
        with pytest.raises(ValueError, match="num_train_epochs must be positive"):
            validate_sft_config(config)


class TestCreateDPOConfig:
    """Tests for create_dpo_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_dpo_config()
        assert config.beta == pytest.approx(0.1)
        assert config.loss_type == "sigmoid"
        assert config.max_length == 512
        assert config.max_prompt_length == 256
        assert config.label_smoothing == pytest.approx(0.0)
        assert config.generate_during_eval is False

    def test_custom_beta(self) -> None:
        """Test custom beta value."""
        config = create_dpo_config(beta=0.2)
        assert config.beta == pytest.approx(0.2)

    def test_custom_loss_type(self) -> None:
        """Test custom loss type."""
        config = create_dpo_config(loss_type="hinge")
        assert config.loss_type == "hinge"

    def test_invalid_beta_raises(self) -> None:
        """Test that invalid beta raises error."""
        with pytest.raises(ValueError, match="beta must be positive"):
            create_dpo_config(beta=0.0)


class TestCreatePPOConfig:
    """Tests for create_ppo_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_ppo_config()
        assert config.learning_rate == pytest.approx(1e-5)
        assert config.batch_size == 128
        assert config.mini_batch_size == 32
        assert config.ppo_epochs == 4

    def test_custom_learning_rate(self) -> None:
        """Test custom learning rate."""
        config = create_ppo_config(learning_rate=2e-5)
        assert config.learning_rate == pytest.approx(2e-5)

    def test_custom_ppo_epochs(self) -> None:
        """Test custom PPO epochs."""
        config = create_ppo_config(ppo_epochs=8)
        assert config.ppo_epochs == 8

    def test_invalid_learning_rate_raises(self) -> None:
        """Test that invalid learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            create_ppo_config(learning_rate=0.0)


class TestCreateSFTConfig:
    """Tests for create_sft_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_sft_config()
        assert config.max_seq_length == 2048
        assert config.packing is True
        assert config.dataset_text_field == "text"
        assert config.num_train_epochs == 3

    def test_custom_max_seq_length(self) -> None:
        """Test custom max sequence length."""
        config = create_sft_config(max_seq_length=4096)
        assert config.max_seq_length == 4096

    def test_custom_packing(self) -> None:
        """Test custom packing setting."""
        config = create_sft_config(packing=False)
        assert config.packing is False

    def test_invalid_max_seq_length_raises(self) -> None:
        """Test that invalid max_seq_length raises error."""
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            create_sft_config(max_seq_length=0)


class TestListTrainingMethods:
    """Tests for list_training_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_training_methods()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_training_methods()
        assert result == sorted(result)

    def test_contains_dpo(self) -> None:
        """Test that dpo is in the list."""
        result = list_training_methods()
        assert "dpo" in result

    def test_contains_ppo(self) -> None:
        """Test that ppo is in the list."""
        result = list_training_methods()
        assert "ppo" in result

    def test_contains_sft(self) -> None:
        """Test that sft is in the list."""
        result = list_training_methods()
        assert "sft" in result


class TestGetRecommendedBeta:
    """Tests for get_recommended_beta function."""

    def test_small_model(self) -> None:
        """Test recommended beta for small models."""
        assert get_recommended_beta("small") == pytest.approx(0.5)

    def test_medium_model(self) -> None:
        """Test recommended beta for medium models."""
        assert get_recommended_beta("medium") == pytest.approx(0.1)

    def test_large_model(self) -> None:
        """Test recommended beta for large models."""
        assert get_recommended_beta("large") == pytest.approx(0.05)

    def test_invalid_size_raises(self) -> None:
        """Test that invalid model size raises error."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_beta("xlarge")


class TestEstimateRewardModelParams:
    """Tests for estimate_reward_model_params function."""

    def test_basic_estimate(self) -> None:
        """Test basic parameter estimation."""
        params = estimate_reward_model_params(7_000_000_000)
        assert params > 7_000_000_000

    def test_with_custom_hidden_size(self) -> None:
        """Test estimation with custom hidden size."""
        params = estimate_reward_model_params(
            7_000_000_000, value_head_hidden_size=1024
        )
        assert params == 7_000_000_000 + 1024 + 1

    def test_zero_base_params_raises(self) -> None:
        """Test that zero base params raises error."""
        with pytest.raises(ValueError, match="base_model_params must be positive"):
            estimate_reward_model_params(0)

    def test_zero_hidden_size_raises(self) -> None:
        """Test that zero hidden size raises error."""
        with pytest.raises(ValueError, match="value_head_hidden_size must be positive"):
            estimate_reward_model_params(1_000_000, value_head_hidden_size=0)


class TestCalculateKLPenalty:
    """Tests for calculate_kl_penalty function."""

    def test_basic_calculation(self) -> None:
        """Test basic KL penalty calculation."""
        penalty = calculate_kl_penalty(-2.0, -2.5, kl_coef=0.1)
        assert penalty == pytest.approx(0.05)

    def test_negative_kl(self) -> None:
        """Test KL penalty when policy is worse."""
        penalty = calculate_kl_penalty(-3.0, -2.0, kl_coef=0.1)
        assert penalty < 0

    def test_zero_kl_coef_raises(self) -> None:
        """Test that zero kl_coef raises error."""
        with pytest.raises(ValueError, match="kl_coef must be positive"):
            calculate_kl_penalty(-2.0, -2.5, kl_coef=0.0)

    def test_negative_kl_coef_raises(self) -> None:
        """Test that negative kl_coef raises error."""
        with pytest.raises(ValueError, match="kl_coef must be positive"):
            calculate_kl_penalty(-2.0, -2.5, kl_coef=-0.1)


class TestGetDefaultRewardPrompt:
    """Tests for get_default_reward_prompt function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = get_default_reward_prompt()
        assert isinstance(result, str)

    def test_contains_helpful(self) -> None:
        """Test that prompt contains 'helpful'."""
        result = get_default_reward_prompt()
        assert "helpful" in result.lower()

    def test_contains_harmless(self) -> None:
        """Test that prompt contains 'harmless'."""
        result = get_default_reward_prompt()
        assert "harmless" in result.lower()

    def test_not_empty(self) -> None:
        """Test that prompt is not empty."""
        result = get_default_reward_prompt()
        assert len(result) > 0


class TestConstants:
    """Tests for module constants."""

    def test_valid_training_methods_not_empty(self) -> None:
        """Test VALID_TRAINING_METHODS is not empty."""
        assert len(VALID_TRAINING_METHODS) > 0

    def test_valid_dpo_loss_types_contents(self) -> None:
        """Test VALID_DPO_LOSS_TYPES contains expected values."""
        assert "sigmoid" in VALID_DPO_LOSS_TYPES
        assert "hinge" in VALID_DPO_LOSS_TYPES
        assert "ipo" in VALID_DPO_LOSS_TYPES
