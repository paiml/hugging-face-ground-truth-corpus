"""Tests for DPO (Direct Preference Optimization) training utilities."""

from __future__ import annotations

import math

import pytest

from hf_gtc.training.dpo import (
    VALID_DPO_VARIANTS,
    VALID_LOSS_TYPES,
    VALID_REFERENCE_POLICIES,
    DPOConfig,
    DPOStats,
    DPOTrainingConfig,
    DPOVariant,
    LossType,
    PreferencePair,
    ReferenceConfig,
    ReferencePolicy,
    calculate_dpo_loss,
    create_dpo_config,
    create_dpo_training_config,
    create_preference_pair,
    create_reference_config,
    estimate_training_steps,
    get_dpo_variant,
    get_loss_type,
    get_reference_policy,
    list_dpo_variants,
    list_loss_types,
    list_reference_policies,
    validate_dpo_config,
    validate_preference_pair,
)


class TestDPOVariant:
    """Tests for DPOVariant enum."""

    def test_standard_value(self) -> None:
        """Test STANDARD value."""
        assert DPOVariant.STANDARD.value == "standard"

    def test_rspo_value(self) -> None:
        """Test RSPO value."""
        assert DPOVariant.RSPO.value == "rspo"

    def test_ipo_value(self) -> None:
        """Test IPO value."""
        assert DPOVariant.IPO.value == "ipo"

    def test_kto_value(self) -> None:
        """Test KTO value."""
        assert DPOVariant.KTO.value == "kto"

    def test_orpo_value(self) -> None:
        """Test ORPO value."""
        assert DPOVariant.ORPO.value == "orpo"

    def test_all_variants_in_valid_set(self) -> None:
        """Test all variants are in VALID_DPO_VARIANTS."""
        for variant in DPOVariant:
            assert variant.value in VALID_DPO_VARIANTS


class TestLossType:
    """Tests for LossType enum."""

    def test_sigmoid_value(self) -> None:
        """Test SIGMOID value."""
        assert LossType.SIGMOID.value == "sigmoid"

    def test_hinge_value(self) -> None:
        """Test HINGE value."""
        assert LossType.HINGE.value == "hinge"

    def test_ipo_loss_value(self) -> None:
        """Test IPO_LOSS value."""
        assert LossType.IPO_LOSS.value == "ipo_loss"

    def test_all_loss_types_in_valid_set(self) -> None:
        """Test all loss types are in VALID_LOSS_TYPES."""
        for loss_type in LossType:
            assert loss_type.value in VALID_LOSS_TYPES


class TestReferencePolicy:
    """Tests for ReferencePolicy enum."""

    def test_frozen_value(self) -> None:
        """Test FROZEN value."""
        assert ReferencePolicy.FROZEN.value == "frozen"

    def test_online_value(self) -> None:
        """Test ONLINE value."""
        assert ReferencePolicy.ONLINE.value == "online"

    def test_self_value(self) -> None:
        """Test SELF value."""
        assert ReferencePolicy.SELF.value == "self"

    def test_all_policies_in_valid_set(self) -> None:
        """Test all reference policies are in VALID_REFERENCE_POLICIES."""
        for policy in ReferencePolicy:
            assert policy.value in VALID_REFERENCE_POLICIES


class TestDPOConfig:
    """Tests for DPOConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating DPOConfig."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=0.1,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=0.0,
        )
        assert config.variant == DPOVariant.STANDARD
        assert config.beta == pytest.approx(0.1)
        assert config.loss_type == LossType.SIGMOID
        assert config.reference_free is False
        assert config.label_smoothing == pytest.approx(0.0)

    def test_ipo_config(self) -> None:
        """Test creating IPO config."""
        config = DPOConfig(
            variant=DPOVariant.IPO,
            beta=0.5,
            loss_type=LossType.IPO_LOSS,
            reference_free=False,
            label_smoothing=0.1,
        )
        assert config.variant == DPOVariant.IPO
        assert config.label_smoothing == pytest.approx(0.1)

    def test_frozen(self) -> None:
        """Test that DPOConfig is immutable."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=0.1,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=0.0,
        )
        with pytest.raises(AttributeError):
            config.beta = 0.2  # type: ignore[misc]


class TestPreferencePair:
    """Tests for PreferencePair dataclass."""

    def test_creation(self) -> None:
        """Test creating PreferencePair."""
        pair = PreferencePair(
            prompt="What is 2+2?",
            chosen="4",
            rejected="5",
            chosen_score=0.9,
            rejected_score=0.1,
        )
        assert pair.prompt == "What is 2+2?"
        assert pair.chosen == "4"
        assert pair.rejected == "5"
        assert pair.chosen_score == pytest.approx(0.9)
        assert pair.rejected_score == pytest.approx(0.1)

    def test_none_scores(self) -> None:
        """Test creating PreferencePair with None scores."""
        pair = PreferencePair(
            prompt="Hello",
            chosen="Hi!",
            rejected="Go away",
            chosen_score=None,
            rejected_score=None,
        )
        assert pair.chosen_score is None
        assert pair.rejected_score is None

    def test_frozen(self) -> None:
        """Test that PreferencePair is immutable."""
        pair = PreferencePair(
            prompt="Q",
            chosen="A",
            rejected="B",
            chosen_score=None,
            rejected_score=None,
        )
        with pytest.raises(AttributeError):
            pair.prompt = "New Q"  # type: ignore[misc]


class TestDPOTrainingConfig:
    """Tests for DPOTrainingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating DPOTrainingConfig."""
        config = DPOTrainingConfig(
            learning_rate=5e-7,
            batch_size=4,
            max_length=512,
            max_prompt_length=256,
        )
        assert config.learning_rate == pytest.approx(5e-7)
        assert config.batch_size == 4
        assert config.max_length == 512
        assert config.max_prompt_length == 256

    def test_different_values(self) -> None:
        """Test creating DPOTrainingConfig with different values."""
        config = DPOTrainingConfig(
            learning_rate=1e-6,
            batch_size=8,
            max_length=1024,
            max_prompt_length=512,
        )
        assert config.learning_rate == pytest.approx(1e-6)
        assert config.max_length == 1024

    def test_frozen(self) -> None:
        """Test that DPOTrainingConfig is immutable."""
        config = DPOTrainingConfig(
            learning_rate=5e-7,
            batch_size=4,
            max_length=512,
            max_prompt_length=256,
        )
        with pytest.raises(AttributeError):
            config.batch_size = 8  # type: ignore[misc]


class TestReferenceConfig:
    """Tests for ReferenceConfig dataclass."""

    def test_frozen_policy_creation(self) -> None:
        """Test creating frozen reference config."""
        config = ReferenceConfig(
            policy_type=ReferencePolicy.FROZEN,
            update_frequency=0,
            ema_decay=0.0,
        )
        assert config.policy_type == ReferencePolicy.FROZEN
        assert config.update_frequency == 0
        assert config.ema_decay == pytest.approx(0.0)

    def test_online_policy_creation(self) -> None:
        """Test creating online reference config."""
        config = ReferenceConfig(
            policy_type=ReferencePolicy.ONLINE,
            update_frequency=100,
            ema_decay=0.99,
        )
        assert config.policy_type == ReferencePolicy.ONLINE
        assert config.update_frequency == 100
        assert config.ema_decay == pytest.approx(0.99)

    def test_frozen(self) -> None:
        """Test that ReferenceConfig is immutable."""
        config = ReferenceConfig(
            policy_type=ReferencePolicy.FROZEN,
            update_frequency=0,
            ema_decay=0.0,
        )
        with pytest.raises(AttributeError):
            config.update_frequency = 100  # type: ignore[misc]


class TestDPOStats:
    """Tests for DPOStats dataclass."""

    def test_creation(self) -> None:
        """Test creating DPOStats."""
        stats = DPOStats(
            chosen_rewards_mean=2.5,
            rejected_rewards_mean=-1.5,
            reward_margin=4.0,
            accuracy=0.85,
        )
        assert stats.chosen_rewards_mean == pytest.approx(2.5)
        assert stats.rejected_rewards_mean == pytest.approx(-1.5)
        assert stats.reward_margin == pytest.approx(4.0)
        assert stats.accuracy == pytest.approx(0.85)

    def test_different_values(self) -> None:
        """Test creating DPOStats with different values."""
        stats = DPOStats(
            chosen_rewards_mean=1.0,
            rejected_rewards_mean=0.5,
            reward_margin=0.5,
            accuracy=0.65,
        )
        assert stats.reward_margin == pytest.approx(0.5)
        assert stats.accuracy == pytest.approx(0.65)

    def test_frozen(self) -> None:
        """Test that DPOStats is immutable."""
        stats = DPOStats(
            chosen_rewards_mean=2.5,
            rejected_rewards_mean=-1.5,
            reward_margin=4.0,
            accuracy=0.85,
        )
        with pytest.raises(AttributeError):
            stats.accuracy = 0.9  # type: ignore[misc]


class TestValidateDPOConfig:
    """Tests for validate_dpo_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=0.1,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=0.0,
        )
        validate_dpo_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_dpo_config(None)  # type: ignore[arg-type]

    def test_zero_beta_raises_error(self) -> None:
        """Test that zero beta raises ValueError."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=0.0,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=0.0,
        )
        with pytest.raises(ValueError, match="beta must be positive"):
            validate_dpo_config(config)

    def test_negative_beta_raises_error(self) -> None:
        """Test that negative beta raises ValueError."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=-0.1,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=0.0,
        )
        with pytest.raises(ValueError, match="beta must be positive"):
            validate_dpo_config(config)

    def test_negative_label_smoothing_raises_error(self) -> None:
        """Test that negative label_smoothing raises ValueError."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=0.1,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=-0.1,
        )
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            validate_dpo_config(config)

    def test_one_label_smoothing_raises_error(self) -> None:
        """Test that label_smoothing=1.0 raises ValueError."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=0.1,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=1.0,
        )
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            validate_dpo_config(config)

    def test_valid_label_smoothing_boundary(self) -> None:
        """Test that label_smoothing=0.99 is valid."""
        config = DPOConfig(
            variant=DPOVariant.STANDARD,
            beta=0.1,
            loss_type=LossType.SIGMOID,
            reference_free=False,
            label_smoothing=0.99,
        )
        validate_dpo_config(config)  # Should not raise


class TestValidatePreferencePair:
    """Tests for validate_preference_pair function."""

    def test_valid_pair(self) -> None:
        """Test validating valid pair."""
        pair = PreferencePair(
            prompt="Question?",
            chosen="Good answer",
            rejected="Bad answer",
            chosen_score=None,
            rejected_score=None,
        )
        validate_preference_pair(pair)  # Should not raise

    def test_none_pair_raises_error(self) -> None:
        """Test that None pair raises ValueError."""
        with pytest.raises(ValueError, match="pair cannot be None"):
            validate_preference_pair(None)  # type: ignore[arg-type]

    def test_empty_prompt_raises_error(self) -> None:
        """Test that empty prompt raises ValueError."""
        pair = PreferencePair(
            prompt="",
            chosen="Good",
            rejected="Bad",
            chosen_score=None,
            rejected_score=None,
        )
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            validate_preference_pair(pair)

    def test_empty_chosen_raises_error(self) -> None:
        """Test that empty chosen raises ValueError."""
        pair = PreferencePair(
            prompt="Q?",
            chosen="",
            rejected="Bad",
            chosen_score=None,
            rejected_score=None,
        )
        with pytest.raises(ValueError, match="chosen cannot be empty"):
            validate_preference_pair(pair)

    def test_empty_rejected_raises_error(self) -> None:
        """Test that empty rejected raises ValueError."""
        pair = PreferencePair(
            prompt="Q?",
            chosen="Good",
            rejected="",
            chosen_score=None,
            rejected_score=None,
        )
        with pytest.raises(ValueError, match="rejected cannot be empty"):
            validate_preference_pair(pair)

    def test_identical_chosen_rejected_raises_error(self) -> None:
        """Test that identical chosen and rejected raises ValueError."""
        pair = PreferencePair(
            prompt="Q?",
            chosen="Same",
            rejected="Same",
            chosen_score=None,
            rejected_score=None,
        )
        with pytest.raises(ValueError, match="chosen and rejected cannot be identical"):
            validate_preference_pair(pair)


class TestCreateDPOConfig:
    """Tests for create_dpo_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating config with defaults."""
        config = create_dpo_config()
        assert config.variant == DPOVariant.STANDARD
        assert config.beta == pytest.approx(0.1)
        assert config.loss_type == LossType.SIGMOID
        assert config.reference_free is False
        assert config.label_smoothing == pytest.approx(0.0)

    def test_custom_beta(self) -> None:
        """Test creating config with custom beta."""
        config = create_dpo_config(beta=0.2)
        assert config.beta == pytest.approx(0.2)

    def test_string_variant(self) -> None:
        """Test creating config with string variant."""
        config = create_dpo_config(variant="ipo")
        assert config.variant == DPOVariant.IPO

    def test_string_loss_type(self) -> None:
        """Test creating config with string loss type."""
        config = create_dpo_config(loss_type="hinge")
        assert config.loss_type == LossType.HINGE

    def test_enum_variant(self) -> None:
        """Test creating config with enum variant."""
        config = create_dpo_config(variant=DPOVariant.KTO)
        assert config.variant == DPOVariant.KTO

    def test_enum_loss_type(self) -> None:
        """Test creating config with enum loss type."""
        config = create_dpo_config(loss_type=LossType.IPO_LOSS)
        assert config.loss_type == LossType.IPO_LOSS

    def test_reference_free(self) -> None:
        """Test creating config with reference_free=True."""
        config = create_dpo_config(reference_free=True)
        assert config.reference_free is True

    def test_label_smoothing(self) -> None:
        """Test creating config with label smoothing."""
        config = create_dpo_config(label_smoothing=0.1)
        assert config.label_smoothing == pytest.approx(0.1)

    def test_zero_beta_raises_error(self) -> None:
        """Test that zero beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            create_dpo_config(beta=0.0)

    def test_invalid_variant_raises_error(self) -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="invalid DPO variant"):
            create_dpo_config(variant="invalid")

    def test_invalid_loss_type_raises_error(self) -> None:
        """Test that invalid loss type raises ValueError."""
        with pytest.raises(ValueError, match="invalid loss type"):
            create_dpo_config(loss_type="invalid")


class TestCreatePreferencePair:
    """Tests for create_preference_pair function."""

    def test_creates_pair(self) -> None:
        """Test creating preference pair."""
        pair = create_preference_pair(
            prompt="What is Python?",
            chosen="A programming language",
            rejected="A snake",
        )
        assert pair.prompt == "What is Python?"
        assert pair.chosen == "A programming language"
        assert pair.rejected == "A snake"
        assert pair.chosen_score is None
        assert pair.rejected_score is None

    def test_with_scores(self) -> None:
        """Test creating preference pair with scores."""
        pair = create_preference_pair(
            prompt="Hello",
            chosen="Hi!",
            rejected="...",
            chosen_score=0.9,
            rejected_score=0.2,
        )
        assert pair.chosen_score == pytest.approx(0.9)
        assert pair.rejected_score == pytest.approx(0.2)

    def test_empty_prompt_raises_error(self) -> None:
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            create_preference_pair(prompt="", chosen="a", rejected="b")

    def test_empty_chosen_raises_error(self) -> None:
        """Test that empty chosen raises ValueError."""
        with pytest.raises(ValueError, match="chosen cannot be empty"):
            create_preference_pair(prompt="q", chosen="", rejected="b")

    def test_empty_rejected_raises_error(self) -> None:
        """Test that empty rejected raises ValueError."""
        with pytest.raises(ValueError, match="rejected cannot be empty"):
            create_preference_pair(prompt="q", chosen="a", rejected="")

    def test_identical_responses_raises_error(self) -> None:
        """Test that identical responses raise ValueError."""
        with pytest.raises(ValueError, match="chosen and rejected cannot be identical"):
            create_preference_pair(prompt="q", chosen="same", rejected="same")


class TestCreateDPOTrainingConfig:
    """Tests for create_dpo_training_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating config with defaults."""
        config = create_dpo_training_config()
        assert config.learning_rate == pytest.approx(5e-7)
        assert config.batch_size == 4
        assert config.max_length == 512
        assert config.max_prompt_length == 256

    def test_custom_learning_rate(self) -> None:
        """Test creating config with custom learning rate."""
        config = create_dpo_training_config(learning_rate=1e-6)
        assert config.learning_rate == pytest.approx(1e-6)

    def test_custom_batch_size(self) -> None:
        """Test creating config with custom batch size."""
        config = create_dpo_training_config(batch_size=8)
        assert config.batch_size == 8

    def test_custom_max_length(self) -> None:
        """Test creating config with custom max length."""
        config = create_dpo_training_config(max_length=1024, max_prompt_length=256)
        assert config.max_length == 1024

    def test_zero_learning_rate_raises_error(self) -> None:
        """Test that zero learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            create_dpo_training_config(learning_rate=0.0)

    def test_negative_learning_rate_raises_error(self) -> None:
        """Test that negative learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            create_dpo_training_config(learning_rate=-1e-6)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_dpo_training_config(batch_size=0)

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_dpo_training_config(batch_size=-1)

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_dpo_training_config(max_length=0)

    def test_zero_max_prompt_length_raises_error(self) -> None:
        """Test that zero max prompt length raises ValueError."""
        with pytest.raises(ValueError, match="max_prompt_length must be positive"):
            create_dpo_training_config(max_prompt_length=0)

    def test_prompt_length_equals_max_length_raises_error(self) -> None:
        """Test that max_prompt_length >= max_length raises ValueError."""
        with pytest.raises(ValueError, match=r"max_prompt_length.*must be less than"):
            create_dpo_training_config(max_length=256, max_prompt_length=256)

    def test_prompt_length_exceeds_max_length_raises_error(self) -> None:
        """Test that max_prompt_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match=r"max_prompt_length.*must be less than"):
            create_dpo_training_config(max_length=256, max_prompt_length=512)


class TestCreateReferenceConfig:
    """Tests for create_reference_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating config with defaults."""
        config = create_reference_config()
        assert config.policy_type == ReferencePolicy.FROZEN
        assert config.update_frequency == 0
        assert config.ema_decay == pytest.approx(0.0)

    def test_string_policy_type(self) -> None:
        """Test creating config with string policy type."""
        config = create_reference_config(
            policy_type="online", update_frequency=100, ema_decay=0.99
        )
        assert config.policy_type == ReferencePolicy.ONLINE

    def test_enum_policy_type(self) -> None:
        """Test creating config with enum policy type."""
        config = create_reference_config(policy_type=ReferencePolicy.SELF)
        assert config.policy_type == ReferencePolicy.SELF

    def test_online_with_update_frequency(self) -> None:
        """Test creating online config with update frequency."""
        config = create_reference_config(
            policy_type=ReferencePolicy.ONLINE,
            update_frequency=100,
            ema_decay=0.99,
        )
        assert config.update_frequency == 100
        assert config.ema_decay == pytest.approx(0.99)

    def test_invalid_policy_type_raises_error(self) -> None:
        """Test that invalid policy type raises ValueError."""
        with pytest.raises(ValueError, match="invalid reference policy"):
            create_reference_config(policy_type="invalid")

    def test_negative_update_frequency_raises_error(self) -> None:
        """Test that negative update frequency raises ValueError."""
        with pytest.raises(ValueError, match="update_frequency cannot be negative"):
            create_reference_config(update_frequency=-1)

    def test_negative_ema_decay_raises_error(self) -> None:
        """Test that negative ema_decay raises ValueError."""
        with pytest.raises(ValueError, match="ema_decay must be in"):
            create_reference_config(ema_decay=-0.1)

    def test_ema_decay_above_one_raises_error(self) -> None:
        """Test that ema_decay > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="ema_decay must be in"):
            create_reference_config(ema_decay=1.1)

    def test_online_zero_update_frequency_raises_error(self) -> None:
        """Test that ONLINE policy with zero update_frequency raises ValueError."""
        with pytest.raises(
            ValueError, match="ONLINE policy requires positive update_frequency"
        ):
            create_reference_config(
                policy_type=ReferencePolicy.ONLINE, update_frequency=0
            )

    def test_frozen_zero_update_frequency_valid(self) -> None:
        """Test that FROZEN policy with zero update_frequency is valid."""
        config = create_reference_config(
            policy_type=ReferencePolicy.FROZEN, update_frequency=0
        )
        assert config.policy_type == ReferencePolicy.FROZEN


class TestListDPOVariants:
    """Tests for list_dpo_variants function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        variants = list_dpo_variants()
        assert isinstance(variants, list)

    def test_contains_expected_variants(self) -> None:
        """Test that list contains expected variants."""
        variants = list_dpo_variants()
        assert "standard" in variants
        assert "ipo" in variants
        assert "kto" in variants
        assert "orpo" in variants
        assert "rspo" in variants

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        variants = list_dpo_variants()
        assert variants == sorted(variants)


class TestListLossTypes:
    """Tests for list_loss_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_loss_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_loss_types()
        assert "sigmoid" in types
        assert "hinge" in types
        assert "ipo_loss" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_loss_types()
        assert types == sorted(types)


class TestListReferencePolicies:
    """Tests for list_reference_policies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        policies = list_reference_policies()
        assert isinstance(policies, list)

    def test_contains_expected_policies(self) -> None:
        """Test that list contains expected policies."""
        policies = list_reference_policies()
        assert "frozen" in policies
        assert "online" in policies
        assert "self" in policies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        policies = list_reference_policies()
        assert policies == sorted(policies)


class TestGetDPOVariant:
    """Tests for get_dpo_variant function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("standard", DPOVariant.STANDARD),
            ("rspo", DPOVariant.RSPO),
            ("ipo", DPOVariant.IPO),
            ("kto", DPOVariant.KTO),
            ("orpo", DPOVariant.ORPO),
        ],
    )
    def test_get_valid_variant(self, name: str, expected: DPOVariant) -> None:
        """Test getting valid variants."""
        assert get_dpo_variant(name) == expected

    def test_invalid_variant_raises_error(self) -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="invalid DPO variant"):
            get_dpo_variant("invalid")

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid DPO variant"):
            get_dpo_variant("")


class TestGetLossType:
    """Tests for get_loss_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sigmoid", LossType.SIGMOID),
            ("hinge", LossType.HINGE),
            ("ipo_loss", LossType.IPO_LOSS),
        ],
    )
    def test_get_valid_loss_type(self, name: str, expected: LossType) -> None:
        """Test getting valid loss types."""
        assert get_loss_type(name) == expected

    def test_invalid_loss_type_raises_error(self) -> None:
        """Test that invalid loss type raises ValueError."""
        with pytest.raises(ValueError, match="invalid loss type"):
            get_loss_type("invalid")

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid loss type"):
            get_loss_type("")


class TestGetReferencePolicy:
    """Tests for get_reference_policy function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("frozen", ReferencePolicy.FROZEN),
            ("online", ReferencePolicy.ONLINE),
            ("self", ReferencePolicy.SELF),
        ],
    )
    def test_get_valid_policy(self, name: str, expected: ReferencePolicy) -> None:
        """Test getting valid reference policies."""
        assert get_reference_policy(name) == expected

    def test_invalid_policy_raises_error(self) -> None:
        """Test that invalid policy raises ValueError."""
        with pytest.raises(ValueError, match="invalid reference policy"):
            get_reference_policy("invalid")

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid reference policy"):
            get_reference_policy("")


class TestCalculateDPOLoss:
    """Tests for calculate_dpo_loss function."""

    def test_sigmoid_loss_positive_margin(self) -> None:
        """Test sigmoid loss with positive margin."""
        loss = calculate_dpo_loss(2.0, -1.0, beta=0.1)
        assert loss >= 0
        assert loss < 1.0

    def test_sigmoid_loss_negative_margin(self) -> None:
        """Test sigmoid loss with negative margin."""
        loss = calculate_dpo_loss(-1.0, 2.0, beta=0.1)
        assert loss > 0

    def test_sigmoid_loss_zero_margin(self) -> None:
        """Test sigmoid loss with zero margin."""
        loss = calculate_dpo_loss(1.0, 1.0, beta=0.1)
        # When margin is 0, loss should be log(2) for sigmoid
        expected = math.log(2)
        assert loss == pytest.approx(expected, rel=1e-6)

    def test_hinge_loss_large_margin(self) -> None:
        """Test hinge loss with large positive margin."""
        loss = calculate_dpo_loss(20.0, -10.0, beta=0.1, loss_type=LossType.HINGE)
        # margin = 0.1 * (20 - (-10)) = 3.0, so max(0, 1-3) = 0
        assert loss == pytest.approx(0.0)

    def test_hinge_loss_small_margin(self) -> None:
        """Test hinge loss with small positive margin."""
        loss = calculate_dpo_loss(1.0, 0.0, beta=0.1, loss_type=LossType.HINGE)
        # margin = 0.1 * 1.0 = 0.1, so max(0, 1-0.1) = 0.9
        assert loss == pytest.approx(0.9)

    def test_hinge_loss_negative_margin(self) -> None:
        """Test hinge loss with negative margin."""
        loss = calculate_dpo_loss(-1.0, 1.0, beta=0.1, loss_type=LossType.HINGE)
        # margin = 0.1 * (-2) = -0.2, so max(0, 1-(-0.2)) = 1.2
        assert loss == pytest.approx(1.2)

    def test_ipo_loss(self) -> None:
        """Test IPO loss."""
        loss = calculate_dpo_loss(0.5, 0.3, beta=0.1, loss_type=LossType.IPO_LOSS)
        # margin = 0.1 * (0.5 - 0.3) = 0.02
        # loss = (0.02 - 1)^2 / 2 = 0.9604 / 2 = 0.4802
        expected = ((0.02 - 1) ** 2) / 2
        assert loss == pytest.approx(expected)

    def test_label_smoothing(self) -> None:
        """Test label smoothing with sigmoid loss."""
        loss_no_smooth = calculate_dpo_loss(2.0, -1.0, beta=0.1, label_smoothing=0.0)
        loss_with_smooth = calculate_dpo_loss(2.0, -1.0, beta=0.1, label_smoothing=0.1)
        # With label smoothing, loss is interpolated towards 0.5
        assert loss_with_smooth != loss_no_smooth

    def test_zero_beta_raises_error(self) -> None:
        """Test that zero beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            calculate_dpo_loss(1.0, 0.0, beta=0.0)

    def test_negative_beta_raises_error(self) -> None:
        """Test that negative beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            calculate_dpo_loss(1.0, 0.0, beta=-0.1)

    def test_invalid_label_smoothing_raises_error(self) -> None:
        """Test that invalid label_smoothing raises ValueError."""
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            calculate_dpo_loss(1.0, 0.0, beta=0.1, label_smoothing=1.0)

    def test_negative_label_smoothing_raises_error(self) -> None:
        """Test that negative label_smoothing raises ValueError."""
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            calculate_dpo_loss(1.0, 0.0, beta=0.1, label_smoothing=-0.1)

    @pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 1.0, 10.0])
    def test_various_beta_values(self, beta: float) -> None:
        """Test with various beta values."""
        loss = calculate_dpo_loss(2.0, -1.0, beta=beta)
        assert loss >= 0

    def test_large_positive_margin_stability(self) -> None:
        """Test numerical stability with large positive margin."""
        loss = calculate_dpo_loss(100.0, -100.0, beta=1.0)
        assert loss >= 0
        assert not math.isnan(loss)
        assert not math.isinf(loss)

    def test_large_negative_margin_stability(self) -> None:
        """Test numerical stability with large negative margin."""
        loss = calculate_dpo_loss(-100.0, 100.0, beta=1.0)
        assert loss >= 0
        assert not math.isnan(loss)
        assert not math.isinf(loss)


class TestEstimateTrainingSteps:
    """Tests for estimate_training_steps function."""

    def test_basic_calculation(self) -> None:
        """Test basic step calculation."""
        steps = estimate_training_steps(1000, batch_size=4, num_epochs=3)
        # 1000 / 4 = 250 steps per epoch, 250 * 3 = 750
        assert steps == 750

    def test_with_accumulation(self) -> None:
        """Test calculation with gradient accumulation."""
        steps = estimate_training_steps(
            1000, batch_size=4, num_epochs=1, gradient_accumulation_steps=2
        )
        # Effective batch = 4 * 2 = 8, 1000 / 8 = 125
        assert steps == 125

    def test_partial_batch_excluded(self) -> None:
        """Test that partial batches are excluded."""
        steps = estimate_training_steps(500, batch_size=8, num_epochs=2)
        # 500 / 8 = 62.5 -> 62 steps per epoch, 62 * 2 = 124
        assert steps == 124

    def test_zero_dataset_size_raises_error(self) -> None:
        """Test that zero dataset size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            estimate_training_steps(0)

    def test_negative_dataset_size_raises_error(self) -> None:
        """Test that negative dataset size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            estimate_training_steps(-100)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_training_steps(1000, batch_size=0)

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_training_steps(1000, batch_size=-4)

    def test_zero_num_epochs_raises_error(self) -> None:
        """Test that zero num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            estimate_training_steps(1000, num_epochs=0)

    def test_negative_num_epochs_raises_error(self) -> None:
        """Test that negative num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            estimate_training_steps(1000, num_epochs=-1)

    def test_zero_gradient_accumulation_raises_error(self) -> None:
        """Test that zero gradient accumulation raises ValueError."""
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps must be positive"
        ):
            estimate_training_steps(1000, gradient_accumulation_steps=0)

    def test_negative_gradient_accumulation_raises_error(self) -> None:
        """Test that negative gradient accumulation raises ValueError."""
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps must be positive"
        ):
            estimate_training_steps(1000, gradient_accumulation_steps=-1)

    @pytest.mark.parametrize(
        ("dataset_size", "batch_size", "num_epochs", "expected"),
        [
            (100, 10, 1, 10),
            (100, 10, 2, 20),
            (1000, 32, 3, 93),  # 1000 // 32 = 31, 31 * 3 = 93
            (50, 8, 1, 6),  # 50 // 8 = 6
        ],
    )
    def test_parametrized_calculations(
        self, dataset_size: int, batch_size: int, num_epochs: int, expected: int
    ) -> None:
        """Test various step calculations."""
        steps = estimate_training_steps(
            dataset_size, batch_size=batch_size, num_epochs=num_epochs
        )
        assert steps == expected

    def test_small_dataset_large_batch(self) -> None:
        """Test with small dataset and large batch."""
        steps = estimate_training_steps(10, batch_size=100, num_epochs=1)
        # 10 // 100 = 0
        assert steps == 0


class TestValidSets:
    """Tests for valid frozensets."""

    def test_valid_dpo_variants_is_frozenset(self) -> None:
        """Test that VALID_DPO_VARIANTS is a frozenset."""
        assert isinstance(VALID_DPO_VARIANTS, frozenset)

    def test_valid_loss_types_is_frozenset(self) -> None:
        """Test that VALID_LOSS_TYPES is a frozenset."""
        assert isinstance(VALID_LOSS_TYPES, frozenset)

    def test_valid_reference_policies_is_frozenset(self) -> None:
        """Test that VALID_REFERENCE_POLICIES is a frozenset."""
        assert isinstance(VALID_REFERENCE_POLICIES, frozenset)

    def test_valid_dpo_variants_count(self) -> None:
        """Test VALID_DPO_VARIANTS has correct count."""
        assert len(VALID_DPO_VARIANTS) == 5

    def test_valid_loss_types_count(self) -> None:
        """Test VALID_LOSS_TYPES has correct count."""
        assert len(VALID_LOSS_TYPES) == 3

    def test_valid_reference_policies_count(self) -> None:
        """Test VALID_REFERENCE_POLICIES has correct count."""
        assert len(VALID_REFERENCE_POLICIES) == 3
