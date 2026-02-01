"""Tests for Proximal Policy Optimization (PPO) training utilities."""

from __future__ import annotations

import pytest

from hf_gtc.training.ppo import (
    VALID_PPO_VARIANTS,
    VALID_REWARD_MODEL_TYPES,
    VALID_VALUE_HEAD_TYPES,
    PPOConfig,
    PPOStats,
    PPOTrainingConfig,
    PPOVariant,
    RewardConfig,
    RewardModelType,
    ValueConfig,
    ValueHeadType,
    _exp_safe,
    calculate_gae,
    calculate_ppo_loss,
    create_ppo_config,
    create_ppo_training_config,
    create_reward_config,
    create_value_config,
    get_ppo_variant,
    get_reward_model_type,
    get_value_head_type,
    list_ppo_variants,
    list_reward_model_types,
    list_value_head_types,
    validate_ppo_config,
    validate_ppo_training_config,
    validate_reward_config,
    validate_value_config,
)


class TestPPOVariant:
    """Tests for PPOVariant enum."""

    def test_standard_value(self) -> None:
        """Test STANDARD value."""
        assert PPOVariant.STANDARD.value == "standard"

    def test_reinforce_value(self) -> None:
        """Test REINFORCE value."""
        assert PPOVariant.REINFORCE.value == "reinforce"

    def test_a2c_value(self) -> None:
        """Test A2C value."""
        assert PPOVariant.A2C.value == "a2c"

    def test_ppo_clip_value(self) -> None:
        """Test PPO_CLIP value."""
        assert PPOVariant.PPO_CLIP.value == "ppo_clip"

    def test_ppo_penalty_value(self) -> None:
        """Test PPO_PENALTY value."""
        assert PPOVariant.PPO_PENALTY.value == "ppo_penalty"

    def test_all_variants_in_valid_set(self) -> None:
        """Test all enum values are in VALID_PPO_VARIANTS."""
        for variant in PPOVariant:
            assert variant.value in VALID_PPO_VARIANTS


class TestRewardModelType:
    """Tests for RewardModelType enum."""

    def test_learned_value(self) -> None:
        """Test LEARNED value."""
        assert RewardModelType.LEARNED.value == "learned"

    def test_rule_based_value(self) -> None:
        """Test RULE_BASED value."""
        assert RewardModelType.RULE_BASED.value == "rule_based"

    def test_hybrid_value(self) -> None:
        """Test HYBRID value."""
        assert RewardModelType.HYBRID.value == "hybrid"

    def test_constitutional_value(self) -> None:
        """Test CONSTITUTIONAL value."""
        assert RewardModelType.CONSTITUTIONAL.value == "constitutional"

    def test_all_types_in_valid_set(self) -> None:
        """Test all enum values are in VALID_REWARD_MODEL_TYPES."""
        for model_type in RewardModelType:
            assert model_type.value in VALID_REWARD_MODEL_TYPES


class TestValueHeadType:
    """Tests for ValueHeadType enum."""

    def test_linear_value(self) -> None:
        """Test LINEAR value."""
        assert ValueHeadType.LINEAR.value == "linear"

    def test_mlp_value(self) -> None:
        """Test MLP value."""
        assert ValueHeadType.MLP.value == "mlp"

    def test_shared_value(self) -> None:
        """Test SHARED value."""
        assert ValueHeadType.SHARED.value == "shared"

    def test_all_types_in_valid_set(self) -> None:
        """Test all enum values are in VALID_VALUE_HEAD_TYPES."""
        for head_type in ValueHeadType:
            assert head_type.value in VALID_VALUE_HEAD_TYPES


class TestPPOConfig:
    """Tests for PPOConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PPOConfig instance."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        assert config.variant == PPOVariant.PPO_CLIP
        assert config.clip_range == pytest.approx(0.2)
        assert config.vf_coef == pytest.approx(0.5)
        assert config.ent_coef == pytest.approx(0.01)
        assert config.max_grad_norm == pytest.approx(0.5)
        assert config.target_kl == pytest.approx(0.02)

    def test_frozen(self) -> None:
        """Test that PPOConfig is immutable."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        with pytest.raises(AttributeError):
            config.clip_range = 0.3  # type: ignore[misc]


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating RewardConfig instance."""
        config = RewardConfig(
            model_type=RewardModelType.LEARNED,
            model_id="OpenAssistant/reward-model-deberta-v3-large-v2",
            normalize_rewards=True,
            clip_rewards=10.0,
        )
        assert config.model_type == RewardModelType.LEARNED
        assert config.model_id == "OpenAssistant/reward-model-deberta-v3-large-v2"
        assert config.normalize_rewards is True
        assert config.clip_rewards == pytest.approx(10.0)

    def test_creation_with_none_model_id(self) -> None:
        """Test creating RewardConfig with None model_id."""
        config = RewardConfig(
            model_type=RewardModelType.RULE_BASED,
            model_id=None,
            normalize_rewards=False,
            clip_rewards=None,
        )
        assert config.model_id is None
        assert config.clip_rewards is None

    def test_frozen(self) -> None:
        """Test that RewardConfig is immutable."""
        config = RewardConfig(
            model_type=RewardModelType.LEARNED,
            model_id="test-model",
            normalize_rewards=True,
            clip_rewards=10.0,
        )
        with pytest.raises(AttributeError):
            config.model_id = "new-model"  # type: ignore[misc]


class TestValueConfig:
    """Tests for ValueConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ValueConfig instance."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
        )
        assert config.head_type == ValueHeadType.MLP
        assert config.hidden_size == 256
        assert config.num_layers == 2
        assert config.dropout == pytest.approx(0.1)

    def test_frozen(self) -> None:
        """Test that ValueConfig is immutable."""
        config = ValueConfig(
            head_type=ValueHeadType.LINEAR,
            hidden_size=128,
            num_layers=1,
            dropout=0.0,
        )
        with pytest.raises(AttributeError):
            config.hidden_size = 256  # type: ignore[misc]


class TestPPOTrainingConfig:
    """Tests for PPOTrainingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PPOTrainingConfig instance."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        assert config.learning_rate == pytest.approx(1e-5)
        assert config.batch_size == 128
        assert config.mini_batch_size == 32
        assert config.ppo_epochs == 4
        assert config.rollout_steps == 128

    def test_frozen(self) -> None:
        """Test that PPOTrainingConfig is immutable."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(AttributeError):
            config.batch_size = 256  # type: ignore[misc]


class TestPPOStats:
    """Tests for PPOStats dataclass."""

    def test_creation(self) -> None:
        """Test creating PPOStats instance."""
        stats = PPOStats(
            policy_loss=0.1,
            value_loss=0.5,
            entropy=0.8,
            kl_divergence=0.01,
            clip_fraction=0.15,
            explained_variance=0.85,
        )
        assert stats.policy_loss == pytest.approx(0.1)
        assert stats.value_loss == pytest.approx(0.5)
        assert stats.entropy == pytest.approx(0.8)
        assert stats.kl_divergence == pytest.approx(0.01)
        assert stats.clip_fraction == pytest.approx(0.15)
        assert stats.explained_variance == pytest.approx(0.85)

    def test_frozen(self) -> None:
        """Test that PPOStats is immutable."""
        stats = PPOStats(
            policy_loss=0.1,
            value_loss=0.5,
            entropy=0.8,
            kl_divergence=0.01,
            clip_fraction=0.15,
            explained_variance=0.85,
        )
        with pytest.raises(AttributeError):
            stats.policy_loss = 0.2  # type: ignore[misc]


class TestValidatePPOConfig:
    """Tests for validate_ppo_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        validate_ppo_config(config)  # Should not raise

    def test_zero_clip_range_raises_error(self) -> None:
        """Test that zero clip_range raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.0,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        with pytest.raises(ValueError, match="clip_range must be positive"):
            validate_ppo_config(config)

    def test_negative_clip_range_raises_error(self) -> None:
        """Test that negative clip_range raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=-0.1,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        with pytest.raises(ValueError, match="clip_range must be positive"):
            validate_ppo_config(config)

    def test_negative_vf_coef_raises_error(self) -> None:
        """Test that negative vf_coef raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=-0.1,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        with pytest.raises(ValueError, match="vf_coef must be non-negative"):
            validate_ppo_config(config)

    def test_zero_vf_coef_valid(self) -> None:
        """Test that zero vf_coef is valid."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.0,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        validate_ppo_config(config)  # Should not raise

    def test_negative_ent_coef_raises_error(self) -> None:
        """Test that negative ent_coef raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=-0.01,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        with pytest.raises(ValueError, match="ent_coef must be non-negative"):
            validate_ppo_config(config)

    def test_zero_ent_coef_valid(self) -> None:
        """Test that zero ent_coef is valid."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            target_kl=0.02,
        )
        validate_ppo_config(config)  # Should not raise

    def test_zero_max_grad_norm_raises_error(self) -> None:
        """Test that zero max_grad_norm raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.0,
            target_kl=0.02,
        )
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            validate_ppo_config(config)

    def test_negative_max_grad_norm_raises_error(self) -> None:
        """Test that negative max_grad_norm raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=-0.5,
            target_kl=0.02,
        )
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            validate_ppo_config(config)

    def test_zero_target_kl_raises_error(self) -> None:
        """Test that zero target_kl raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.0,
        )
        with pytest.raises(ValueError, match="target_kl must be positive"):
            validate_ppo_config(config)

    def test_negative_target_kl_raises_error(self) -> None:
        """Test that negative target_kl raises ValueError."""
        config = PPOConfig(
            variant=PPOVariant.PPO_CLIP,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=-0.02,
        )
        with pytest.raises(ValueError, match="target_kl must be positive"):
            validate_ppo_config(config)


class TestValidateRewardConfig:
    """Tests for validate_reward_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = RewardConfig(
            model_type=RewardModelType.LEARNED,
            model_id="test-model",
            normalize_rewards=True,
            clip_rewards=10.0,
        )
        validate_reward_config(config)  # Should not raise

    def test_learned_without_model_id_raises_error(self) -> None:
        """Test that learned model without model_id raises ValueError."""
        config = RewardConfig(
            model_type=RewardModelType.LEARNED,
            model_id=None,
            normalize_rewards=True,
            clip_rewards=10.0,
        )
        with pytest.raises(ValueError, match="model_id is required for learned"):
            validate_reward_config(config)

    def test_learned_with_empty_model_id_raises_error(self) -> None:
        """Test that learned model with empty model_id raises ValueError."""
        config = RewardConfig(
            model_type=RewardModelType.LEARNED,
            model_id="",
            normalize_rewards=True,
            clip_rewards=10.0,
        )
        with pytest.raises(ValueError, match="model_id is required for learned"):
            validate_reward_config(config)

    def test_rule_based_without_model_id_valid(self) -> None:
        """Test that rule_based model without model_id is valid."""
        config = RewardConfig(
            model_type=RewardModelType.RULE_BASED,
            model_id=None,
            normalize_rewards=True,
            clip_rewards=10.0,
        )
        validate_reward_config(config)  # Should not raise

    def test_zero_clip_rewards_raises_error(self) -> None:
        """Test that zero clip_rewards raises ValueError."""
        config = RewardConfig(
            model_type=RewardModelType.RULE_BASED,
            model_id=None,
            normalize_rewards=True,
            clip_rewards=0.0,
        )
        with pytest.raises(ValueError, match="clip_rewards must be positive"):
            validate_reward_config(config)

    def test_negative_clip_rewards_raises_error(self) -> None:
        """Test that negative clip_rewards raises ValueError."""
        config = RewardConfig(
            model_type=RewardModelType.RULE_BASED,
            model_id=None,
            normalize_rewards=True,
            clip_rewards=-5.0,
        )
        with pytest.raises(ValueError, match="clip_rewards must be positive"):
            validate_reward_config(config)

    def test_none_clip_rewards_valid(self) -> None:
        """Test that None clip_rewards is valid."""
        config = RewardConfig(
            model_type=RewardModelType.RULE_BASED,
            model_id=None,
            normalize_rewards=True,
            clip_rewards=None,
        )
        validate_reward_config(config)  # Should not raise


class TestValidateValueConfig:
    """Tests for validate_value_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
        )
        validate_value_config(config)  # Should not raise

    def test_zero_hidden_size_raises_error(self) -> None:
        """Test that zero hidden_size raises ValueError."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=0,
            num_layers=2,
            dropout=0.1,
        )
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_value_config(config)

    def test_negative_hidden_size_raises_error(self) -> None:
        """Test that negative hidden_size raises ValueError."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=-128,
            num_layers=2,
            dropout=0.1,
        )
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_value_config(config)

    def test_zero_num_layers_raises_error(self) -> None:
        """Test that zero num_layers raises ValueError."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=0,
            dropout=0.1,
        )
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_value_config(config)

    def test_negative_num_layers_raises_error(self) -> None:
        """Test that negative num_layers raises ValueError."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=-1,
            dropout=0.1,
        )
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_value_config(config)

    def test_negative_dropout_raises_error(self) -> None:
        """Test that negative dropout raises ValueError."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=2,
            dropout=-0.1,
        )
        with pytest.raises(ValueError, match="dropout must be in"):
            validate_value_config(config)

    def test_dropout_one_raises_error(self) -> None:
        """Test that dropout=1.0 raises ValueError."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=2,
            dropout=1.0,
        )
        with pytest.raises(ValueError, match="dropout must be in"):
            validate_value_config(config)

    def test_dropout_greater_than_one_raises_error(self) -> None:
        """Test that dropout>1.0 raises ValueError."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=2,
            dropout=1.5,
        )
        with pytest.raises(ValueError, match="dropout must be in"):
            validate_value_config(config)

    def test_zero_dropout_valid(self) -> None:
        """Test that zero dropout is valid."""
        config = ValueConfig(
            head_type=ValueHeadType.MLP,
            hidden_size=256,
            num_layers=2,
            dropout=0.0,
        )
        validate_value_config(config)  # Should not raise


class TestValidatePPOTrainingConfig:
    """Tests for validate_ppo_training_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        validate_ppo_training_config(config)  # Should not raise

    def test_zero_learning_rate_raises_error(self) -> None:
        """Test that zero learning_rate raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=0.0,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_ppo_training_config(config)

    def test_negative_learning_rate_raises_error(self) -> None:
        """Test that negative learning_rate raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=-1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_ppo_training_config(config)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=0,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_ppo_training_config(config)

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch_size raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=-128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_ppo_training_config(config)

    def test_zero_mini_batch_size_raises_error(self) -> None:
        """Test that zero mini_batch_size raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=0,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="mini_batch_size must be positive"):
            validate_ppo_training_config(config)

    def test_negative_mini_batch_size_raises_error(self) -> None:
        """Test that negative mini_batch_size raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=-32,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="mini_batch_size must be positive"):
            validate_ppo_training_config(config)

    def test_mini_batch_exceeds_batch_raises_error(self) -> None:
        """Test that mini_batch_size > batch_size raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=32,
            mini_batch_size=64,
            ppo_epochs=4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match=r"mini_batch_size.*cannot exceed"):
            validate_ppo_training_config(config)

    def test_mini_batch_equals_batch_valid(self) -> None:
        """Test that mini_batch_size == batch_size is valid."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=64,
            mini_batch_size=64,
            ppo_epochs=4,
            rollout_steps=128,
        )
        validate_ppo_training_config(config)  # Should not raise

    def test_zero_ppo_epochs_raises_error(self) -> None:
        """Test that zero ppo_epochs raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=0,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="ppo_epochs must be positive"):
            validate_ppo_training_config(config)

    def test_negative_ppo_epochs_raises_error(self) -> None:
        """Test that negative ppo_epochs raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=-4,
            rollout_steps=128,
        )
        with pytest.raises(ValueError, match="ppo_epochs must be positive"):
            validate_ppo_training_config(config)

    def test_zero_rollout_steps_raises_error(self) -> None:
        """Test that zero rollout_steps raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=0,
        )
        with pytest.raises(ValueError, match="rollout_steps must be positive"):
            validate_ppo_training_config(config)

    def test_negative_rollout_steps_raises_error(self) -> None:
        """Test that negative rollout_steps raises ValueError."""
        config = PPOTrainingConfig(
            learning_rate=1e-5,
            batch_size=128,
            mini_batch_size=32,
            ppo_epochs=4,
            rollout_steps=-128,
        )
        with pytest.raises(ValueError, match="rollout_steps must be positive"):
            validate_ppo_training_config(config)


class TestCreatePPOConfig:
    """Tests for create_ppo_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_ppo_config()
        assert config.variant == PPOVariant.PPO_CLIP
        assert config.clip_range == pytest.approx(0.2)
        assert config.vf_coef == pytest.approx(0.5)
        assert config.ent_coef == pytest.approx(0.01)
        assert config.max_grad_norm == pytest.approx(0.5)
        assert config.target_kl == pytest.approx(0.02)

    def test_custom_variant(self) -> None:
        """Test with custom variant."""
        config = create_ppo_config(variant="standard")
        assert config.variant == PPOVariant.STANDARD

    @pytest.mark.parametrize(
        "variant",
        ["standard", "reinforce", "a2c", "ppo_clip", "ppo_penalty"],
    )
    def test_all_valid_variants(self, variant: str) -> None:
        """Test all valid variants are accepted."""
        config = create_ppo_config(variant=variant)
        assert config.variant.value == variant

    def test_custom_clip_range(self) -> None:
        """Test with custom clip_range."""
        config = create_ppo_config(clip_range=0.1)
        assert config.clip_range == pytest.approx(0.1)

    def test_custom_vf_coef(self) -> None:
        """Test with custom vf_coef."""
        config = create_ppo_config(vf_coef=0.25)
        assert config.vf_coef == pytest.approx(0.25)

    def test_custom_ent_coef(self) -> None:
        """Test with custom ent_coef."""
        config = create_ppo_config(ent_coef=0.02)
        assert config.ent_coef == pytest.approx(0.02)

    def test_custom_max_grad_norm(self) -> None:
        """Test with custom max_grad_norm."""
        config = create_ppo_config(max_grad_norm=1.0)
        assert config.max_grad_norm == pytest.approx(1.0)

    def test_custom_target_kl(self) -> None:
        """Test with custom target_kl."""
        config = create_ppo_config(target_kl=0.05)
        assert config.target_kl == pytest.approx(0.05)

    def test_invalid_variant_raises_error(self) -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="variant must be one of"):
            create_ppo_config(variant="invalid")

    def test_invalid_clip_range_raises_error(self) -> None:
        """Test that invalid clip_range raises ValueError."""
        with pytest.raises(ValueError, match="clip_range must be positive"):
            create_ppo_config(clip_range=-0.1)


class TestCreateRewardConfig:
    """Tests for create_reward_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_reward_config(
            model_id="test-model",
        )
        assert config.model_type == RewardModelType.LEARNED
        assert config.model_id == "test-model"
        assert config.normalize_rewards is True
        assert config.clip_rewards == pytest.approx(10.0)

    def test_custom_model_type(self) -> None:
        """Test with custom model_type."""
        config = create_reward_config(model_type="rule_based")
        assert config.model_type == RewardModelType.RULE_BASED

    @pytest.mark.parametrize(
        "model_type",
        ["learned", "rule_based", "hybrid", "constitutional"],
    )
    def test_all_valid_model_types(self, model_type: str) -> None:
        """Test all valid model types are accepted."""
        model_id = "test-model" if model_type == "learned" else None
        config = create_reward_config(model_type=model_type, model_id=model_id)
        assert config.model_type.value == model_type

    def test_normalize_rewards_false(self) -> None:
        """Test with normalize_rewards=False."""
        config = create_reward_config(
            model_type="rule_based",
            normalize_rewards=False,
        )
        assert config.normalize_rewards is False

    def test_clip_rewards_none(self) -> None:
        """Test with clip_rewards=None."""
        config = create_reward_config(
            model_type="rule_based",
            clip_rewards=None,
        )
        assert config.clip_rewards is None

    def test_invalid_model_type_raises_error(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_reward_config(model_type="invalid")

    def test_learned_without_model_id_raises_error(self) -> None:
        """Test that learned model without model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id is required"):
            create_reward_config(model_type="learned", model_id=None)


class TestCreateValueConfig:
    """Tests for create_value_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_value_config()
        assert config.head_type == ValueHeadType.MLP
        assert config.hidden_size == 256
        assert config.num_layers == 2
        assert config.dropout == pytest.approx(0.1)

    def test_custom_head_type(self) -> None:
        """Test with custom head_type."""
        config = create_value_config(head_type="linear")
        assert config.head_type == ValueHeadType.LINEAR

    @pytest.mark.parametrize(
        "head_type",
        ["linear", "mlp", "shared"],
    )
    def test_all_valid_head_types(self, head_type: str) -> None:
        """Test all valid head types are accepted."""
        config = create_value_config(head_type=head_type)
        assert config.head_type.value == head_type

    def test_custom_hidden_size(self) -> None:
        """Test with custom hidden_size."""
        config = create_value_config(hidden_size=512)
        assert config.hidden_size == 512

    def test_custom_num_layers(self) -> None:
        """Test with custom num_layers."""
        config = create_value_config(num_layers=3)
        assert config.num_layers == 3

    def test_custom_dropout(self) -> None:
        """Test with custom dropout."""
        config = create_value_config(dropout=0.2)
        assert config.dropout == pytest.approx(0.2)

    def test_invalid_head_type_raises_error(self) -> None:
        """Test that invalid head_type raises ValueError."""
        with pytest.raises(ValueError, match="head_type must be one of"):
            create_value_config(head_type="invalid")

    def test_invalid_hidden_size_raises_error(self) -> None:
        """Test that invalid hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            create_value_config(hidden_size=0)


class TestCreatePPOTrainingConfig:
    """Tests for create_ppo_training_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_ppo_training_config()
        assert config.learning_rate == pytest.approx(1e-5)
        assert config.batch_size == 128
        assert config.mini_batch_size == 32
        assert config.ppo_epochs == 4
        assert config.rollout_steps == 128

    def test_custom_learning_rate(self) -> None:
        """Test with custom learning_rate."""
        config = create_ppo_training_config(learning_rate=2e-5)
        assert config.learning_rate == pytest.approx(2e-5)

    def test_custom_batch_size(self) -> None:
        """Test with custom batch_size."""
        config = create_ppo_training_config(batch_size=256)
        assert config.batch_size == 256

    def test_custom_mini_batch_size(self) -> None:
        """Test with custom mini_batch_size."""
        config = create_ppo_training_config(mini_batch_size=64)
        assert config.mini_batch_size == 64

    def test_custom_ppo_epochs(self) -> None:
        """Test with custom ppo_epochs."""
        config = create_ppo_training_config(ppo_epochs=8)
        assert config.ppo_epochs == 8

    def test_custom_rollout_steps(self) -> None:
        """Test with custom rollout_steps."""
        config = create_ppo_training_config(rollout_steps=256)
        assert config.rollout_steps == 256

    def test_invalid_learning_rate_raises_error(self) -> None:
        """Test that invalid learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            create_ppo_training_config(learning_rate=0.0)

    def test_mini_batch_exceeds_batch_raises_error(self) -> None:
        """Test that mini_batch_size > batch_size raises ValueError."""
        with pytest.raises(ValueError, match=r"mini_batch_size.*cannot exceed"):
            create_ppo_training_config(batch_size=32, mini_batch_size=64)


class TestListPPOVariants:
    """Tests for list_ppo_variants function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        variants = list_ppo_variants()
        assert isinstance(variants, list)

    def test_contains_expected_variants(self) -> None:
        """Test that list contains expected variants."""
        variants = list_ppo_variants()
        assert "ppo_clip" in variants
        assert "standard" in variants
        assert "reinforce" in variants

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        variants = list_ppo_variants()
        assert variants == sorted(variants)

    def test_correct_count(self) -> None:
        """Test correct number of variants."""
        variants = list_ppo_variants()
        assert len(variants) == 5


class TestListRewardModelTypes:
    """Tests for list_reward_model_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_reward_model_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_reward_model_types()
        assert "learned" in types
        assert "constitutional" in types
        assert "rule_based" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_reward_model_types()
        assert types == sorted(types)

    def test_correct_count(self) -> None:
        """Test correct number of types."""
        types = list_reward_model_types()
        assert len(types) == 4


class TestListValueHeadTypes:
    """Tests for list_value_head_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_value_head_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_value_head_types()
        assert "mlp" in types
        assert "linear" in types
        assert "shared" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_value_head_types()
        assert types == sorted(types)

    def test_correct_count(self) -> None:
        """Test correct number of types."""
        types = list_value_head_types()
        assert len(types) == 3


class TestGetPPOVariant:
    """Tests for get_ppo_variant function."""

    def test_get_ppo_clip(self) -> None:
        """Test getting PPO_CLIP variant."""
        assert get_ppo_variant("ppo_clip") == PPOVariant.PPO_CLIP

    def test_get_standard(self) -> None:
        """Test getting STANDARD variant."""
        assert get_ppo_variant("standard") == PPOVariant.STANDARD

    def test_get_reinforce(self) -> None:
        """Test getting REINFORCE variant."""
        assert get_ppo_variant("reinforce") == PPOVariant.REINFORCE

    def test_get_a2c(self) -> None:
        """Test getting A2C variant."""
        assert get_ppo_variant("a2c") == PPOVariant.A2C

    def test_get_ppo_penalty(self) -> None:
        """Test getting PPO_PENALTY variant."""
        assert get_ppo_variant("ppo_penalty") == PPOVariant.PPO_PENALTY

    def test_invalid_raises_error(self) -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="variant must be one of"):
            get_ppo_variant("invalid")


class TestGetRewardModelType:
    """Tests for get_reward_model_type function."""

    def test_get_learned(self) -> None:
        """Test getting LEARNED type."""
        assert get_reward_model_type("learned") == RewardModelType.LEARNED

    def test_get_rule_based(self) -> None:
        """Test getting RULE_BASED type."""
        assert get_reward_model_type("rule_based") == RewardModelType.RULE_BASED

    def test_get_hybrid(self) -> None:
        """Test getting HYBRID type."""
        assert get_reward_model_type("hybrid") == RewardModelType.HYBRID

    def test_get_constitutional(self) -> None:
        """Test getting CONSTITUTIONAL type."""
        assert get_reward_model_type("constitutional") == RewardModelType.CONSTITUTIONAL

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_reward_model_type("invalid")


class TestGetValueHeadType:
    """Tests for get_value_head_type function."""

    def test_get_linear(self) -> None:
        """Test getting LINEAR type."""
        assert get_value_head_type("linear") == ValueHeadType.LINEAR

    def test_get_mlp(self) -> None:
        """Test getting MLP type."""
        assert get_value_head_type("mlp") == ValueHeadType.MLP

    def test_get_shared(self) -> None:
        """Test getting SHARED type."""
        assert get_value_head_type("shared") == ValueHeadType.SHARED

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="head_type must be one of"):
            get_value_head_type("invalid")


class TestCalculateGAE:
    """Tests for calculate_gae function."""

    def test_basic_calculation(self) -> None:
        """Test basic GAE calculation."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.6, 0.7]
        advantages = calculate_gae(rewards, values, next_value=0.8)
        assert len(advantages) == 3
        assert all(isinstance(a, float) for a in advantages)

    def test_single_step(self) -> None:
        """Test single step GAE."""
        # delta = r + gamma * V(s') - V(s)
        # delta = 1.0 + 1.0 * 0.0 - 0.0 = 1.0
        advantages = calculate_gae([1.0], [0.0], 0.0, gamma=1.0, lam=1.0)
        assert advantages[0] == pytest.approx(1.0)

    def test_multiple_steps_gamma_one(self) -> None:
        """Test multiple steps with gamma=1.0."""
        # With gamma=1, lam=1, GAE reduces to sum of TD errors
        rewards = [1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0]
        advantages = calculate_gae(rewards, values, next_value=0.0, gamma=1.0, lam=1.0)
        # A[2] = 1.0, A[1] = 1.0 + 1.0 = 2.0, A[0] = 1.0 + 2.0 = 3.0
        assert advantages[2] == pytest.approx(1.0)
        assert advantages[1] == pytest.approx(2.0)
        assert advantages[0] == pytest.approx(3.0)

    def test_gamma_zero(self) -> None:
        """Test with gamma=0 (no discounting)."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 0.5, 0.5]
        # With gamma=0, delta = r - V(s)
        advantages = calculate_gae(rewards, values, next_value=0.0, gamma=0.0, lam=0.95)
        assert advantages[0] == pytest.approx(0.5)  # 1.0 - 0.5
        assert advantages[1] == pytest.approx(1.5)  # 2.0 - 0.5
        assert advantages[2] == pytest.approx(2.5)  # 3.0 - 0.5

    def test_lam_zero(self) -> None:
        """Test with lam=0 (one-step TD)."""
        rewards = [1.0, 1.0]
        values = [0.0, 0.0]
        # With lam=0, each advantage is just the TD error
        # A[1] = 1.0 + 0.99*0.0 - 0.0 = 1.0
        # A[0] = 1.0 + 0.99*0.0 - 0.0 = 1.0
        advantages = calculate_gae(rewards, values, next_value=0.0, gamma=0.99, lam=0.0)
        assert advantages[0] == pytest.approx(1.0)
        assert advantages[1] == pytest.approx(1.0)

    def test_empty_rewards_raises_error(self) -> None:
        """Test that empty rewards raises ValueError."""
        with pytest.raises(ValueError, match="rewards sequence cannot be empty"):
            calculate_gae([], [], 0.0)

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="rewards and values must have same"):
            calculate_gae([1.0], [0.5, 0.6], 0.0)

    def test_length_mismatch_other_direction(self) -> None:
        """Test length mismatch with values shorter."""
        with pytest.raises(ValueError, match="rewards and values must have same"):
            calculate_gae([1.0, 2.0], [0.5], 0.0)

    def test_gamma_out_of_range_negative_raises_error(self) -> None:
        """Test that gamma < 0 raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be in"):
            calculate_gae([1.0], [0.0], 0.0, gamma=-0.1)

    def test_gamma_out_of_range_above_one_raises_error(self) -> None:
        """Test that gamma > 1 raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be in"):
            calculate_gae([1.0], [0.0], 0.0, gamma=1.1)

    def test_lam_out_of_range_negative_raises_error(self) -> None:
        """Test that lam < 0 raises ValueError."""
        with pytest.raises(ValueError, match="lam must be in"):
            calculate_gae([1.0], [0.0], 0.0, lam=-0.1)

    def test_lam_out_of_range_above_one_raises_error(self) -> None:
        """Test that lam > 1 raises ValueError."""
        with pytest.raises(ValueError, match="lam must be in"):
            calculate_gae([1.0], [0.0], 0.0, lam=1.1)

    def test_boundary_gamma_one(self) -> None:
        """Test gamma=1 is valid."""
        advantages = calculate_gae([1.0], [0.0], 0.0, gamma=1.0)
        assert len(advantages) == 1

    def test_boundary_lam_one(self) -> None:
        """Test lam=1 is valid."""
        advantages = calculate_gae([1.0], [0.0], 0.0, lam=1.0)
        assert len(advantages) == 1


class TestCalculatePPOLoss:
    """Tests for calculate_ppo_loss function."""

    def test_no_change_policy(self) -> None:
        """Test loss when policy hasn't changed."""
        log_probs = [-1.0, -1.5, -2.0]
        old_log_probs = [-1.0, -1.5, -2.0]
        advantages = [1.0, 0.5, -0.5]
        loss, clip_frac = calculate_ppo_loss(log_probs, old_log_probs, advantages)
        # ratio = 1 for all, so loss = -mean(advantages) = -(1.0 + 0.5 - 0.5)/3
        assert loss == pytest.approx(-0.3333, rel=1e-3)
        assert clip_frac == pytest.approx(0.0)

    def test_clipping_occurs(self) -> None:
        """Test that clipping occurs when ratio is outside range."""
        # Large positive log_prob change = ratio >> 1
        log_probs = [-0.5]
        old_log_probs = [-1.5]
        advantages = [1.0]
        _loss, clip_frac = calculate_ppo_loss(
            log_probs, old_log_probs, advantages, clip_range=0.2
        )
        # ratio = exp(1.0) is approximately 2.718, clipped to 1.2
        assert clip_frac == pytest.approx(1.0)

    def test_empty_input_raises_error(self) -> None:
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="input sequences cannot be empty"):
            calculate_ppo_loss([], [], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_ppo_loss([1.0], [1.0], [1.0, 2.0])

    def test_zero_clip_range_raises_error(self) -> None:
        """Test that zero clip_range raises ValueError."""
        with pytest.raises(ValueError, match="clip_range must be positive"):
            calculate_ppo_loss([1.0], [1.0], [1.0], clip_range=0.0)

    def test_negative_clip_range_raises_error(self) -> None:
        """Test that negative clip_range raises ValueError."""
        with pytest.raises(ValueError, match="clip_range must be positive"):
            calculate_ppo_loss([1.0], [1.0], [1.0], clip_range=-0.2)

    def test_negative_advantage_clipping(self) -> None:
        """Test clipping with negative advantage."""
        # With negative advantage, we want to clip low ratios
        log_probs = [-2.5]
        old_log_probs = [-1.5]
        advantages = [-1.0]
        _loss, clip_frac = calculate_ppo_loss(
            log_probs, old_log_probs, advantages, clip_range=0.2
        )
        # ratio = exp(-1.0) is approximately 0.368, clipped to 0.8
        assert clip_frac == pytest.approx(1.0)

    def test_custom_clip_range(self) -> None:
        """Test with custom clip range."""
        log_probs = [-1.0]
        old_log_probs = [-1.0]
        advantages = [1.0]
        loss, clip_frac = calculate_ppo_loss(
            log_probs, old_log_probs, advantages, clip_range=0.1
        )
        assert loss == pytest.approx(-1.0)
        assert clip_frac == pytest.approx(0.0)


class TestExpSafe:
    """Tests for _exp_safe helper function."""

    def test_zero_input(self) -> None:
        """Test exp(0) = 1."""
        assert _exp_safe(0.0) == pytest.approx(1.0)

    def test_small_positive(self) -> None:
        """Test small positive input."""
        import math

        assert _exp_safe(1.0) == pytest.approx(math.exp(1.0))

    def test_small_negative(self) -> None:
        """Test small negative input."""
        import math

        assert _exp_safe(-1.0) == pytest.approx(math.exp(-1.0))

    def test_large_positive_clamped(self) -> None:
        """Test large positive input is clamped."""
        result = _exp_safe(100.0)
        assert result < float("inf")
        import math

        assert result == pytest.approx(math.exp(20.0))

    def test_large_negative_clamped(self) -> None:
        """Test large negative input is clamped."""
        result = _exp_safe(-100.0)
        assert result > 0
        import math

        assert result == pytest.approx(math.exp(-20.0))

    def test_boundary_positive(self) -> None:
        """Test at positive boundary."""
        import math

        assert _exp_safe(20.0) == pytest.approx(math.exp(20.0))

    def test_boundary_negative(self) -> None:
        """Test at negative boundary."""
        import math

        assert _exp_safe(-20.0) == pytest.approx(math.exp(-20.0))
