"""Tests for training.moe module."""

from __future__ import annotations

import math

import pytest

from hf_gtc.training.moe import (
    VALID_EXPERT_ACTIVATIONS,
    VALID_LOAD_BALANCING_LOSSES,
    VALID_ROUTER_TYPES,
    ExpertActivation,
    ExpertConfig,
    LoadBalanceConfig,
    LoadBalancingLoss,
    MoEConfig,
    MoEStats,
    RouterConfig,
    RouterType,
    calculate_capacity_factor,
    calculate_load_balance_loss,
    calculate_router_entropy,
    create_expert_config,
    create_load_balance_config,
    create_moe_config,
    create_moe_stats,
    create_router_config,
    estimate_expert_utilization,
    format_moe_stats,
    get_expert_activation,
    get_load_balancing_loss,
    get_recommended_moe_config,
    get_router_type,
    list_expert_activations,
    list_load_balancing_losses,
    list_router_types,
    validate_expert_config,
    validate_load_balance_config,
    validate_moe_config,
    validate_router_config,
)


class TestRouterType:
    """Tests for RouterType enum."""

    def test_all_types_have_values(self) -> None:
        """All router types have string values."""
        for rt in RouterType:
            assert isinstance(rt.value, str)

    def test_top_k_value(self) -> None:
        """TOP_K has correct value."""
        assert RouterType.TOP_K.value == "top_k"

    def test_expert_choice_value(self) -> None:
        """EXPERT_CHOICE has correct value."""
        assert RouterType.EXPERT_CHOICE.value == "expert_choice"

    def test_soft_value(self) -> None:
        """SOFT has correct value."""
        assert RouterType.SOFT.value == "soft"

    def test_hash_value(self) -> None:
        """HASH has correct value."""
        assert RouterType.HASH.value == "hash"

    def test_valid_router_types_frozenset(self) -> None:
        """VALID_ROUTER_TYPES is a frozenset."""
        assert isinstance(VALID_ROUTER_TYPES, frozenset)
        assert len(VALID_ROUTER_TYPES) == 4


class TestLoadBalancingLoss:
    """Tests for LoadBalancingLoss enum."""

    def test_all_losses_have_values(self) -> None:
        """All load balancing losses have string values."""
        for lb in LoadBalancingLoss:
            assert isinstance(lb.value, str)

    def test_auxiliary_value(self) -> None:
        """AUXILIARY has correct value."""
        assert LoadBalancingLoss.AUXILIARY.value == "auxiliary"

    def test_z_loss_value(self) -> None:
        """Z_LOSS has correct value."""
        assert LoadBalancingLoss.Z_LOSS.value == "z_loss"

    def test_switch_value(self) -> None:
        """SWITCH has correct value."""
        assert LoadBalancingLoss.SWITCH.value == "switch"

    def test_none_value(self) -> None:
        """NONE has correct value."""
        assert LoadBalancingLoss.NONE.value == "none"

    def test_valid_load_balancing_losses_frozenset(self) -> None:
        """VALID_LOAD_BALANCING_LOSSES is a frozenset."""
        assert isinstance(VALID_LOAD_BALANCING_LOSSES, frozenset)
        assert len(VALID_LOAD_BALANCING_LOSSES) == 4


class TestExpertActivation:
    """Tests for ExpertActivation enum."""

    def test_all_activations_have_values(self) -> None:
        """All activations have string values."""
        for ea in ExpertActivation:
            assert isinstance(ea.value, str)

    def test_relu_value(self) -> None:
        """RELU has correct value."""
        assert ExpertActivation.RELU.value == "relu"

    def test_gelu_value(self) -> None:
        """GELU has correct value."""
        assert ExpertActivation.GELU.value == "gelu"

    def test_swiglu_value(self) -> None:
        """SWIGLU has correct value."""
        assert ExpertActivation.SWIGLU.value == "swiglu"

    def test_geglu_value(self) -> None:
        """GEGLU has correct value."""
        assert ExpertActivation.GEGLU.value == "geglu"

    def test_valid_expert_activations_frozenset(self) -> None:
        """VALID_EXPERT_ACTIVATIONS is a frozenset."""
        assert isinstance(VALID_EXPERT_ACTIVATIONS, frozenset)
        assert len(VALID_EXPERT_ACTIVATIONS) == 4


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""

    def test_create_config(self) -> None:
        """Create router config."""
        config = RouterConfig(
            router_type=RouterType.TOP_K,
            num_experts=8,
            top_k=2,
            jitter_noise=0.1,
            temperature=1.0,
        )
        assert config.router_type == RouterType.TOP_K
        assert config.num_experts == 8
        assert config.top_k == 2

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 1.0)
        with pytest.raises(AttributeError):
            config.num_experts = 16  # type: ignore[misc]


class TestLoadBalanceConfig:
    """Tests for LoadBalanceConfig dataclass."""

    def test_create_config(self) -> None:
        """Create load balance config."""
        config = LoadBalanceConfig(
            loss_type=LoadBalancingLoss.AUXILIARY,
            loss_weight=0.01,
            capacity_factor=1.25,
            drop_tokens=False,
        )
        assert config.loss_type == LoadBalancingLoss.AUXILIARY
        assert config.loss_weight == 0.01
        assert config.capacity_factor == 1.25

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        with pytest.raises(AttributeError):
            config.loss_weight = 0.02  # type: ignore[misc]


class TestExpertConfig:
    """Tests for ExpertConfig dataclass."""

    def test_create_config(self) -> None:
        """Create expert config."""
        config = ExpertConfig(
            hidden_dim=4096,
            activation=ExpertActivation.GELU,
            dropout=0.1,
            shared_expert=False,
        )
        assert config.hidden_dim == 4096
        assert config.activation == ExpertActivation.GELU
        assert config.dropout == 0.1

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        with pytest.raises(AttributeError):
            config.hidden_dim = 8192  # type: ignore[misc]


class TestMoEConfig:
    """Tests for MoEConfig dataclass."""

    def test_create_config(self) -> None:
        """Create MoE config."""
        router = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 1.0)
        expert = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        balance = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        config = MoEConfig(
            router_config=router,
            expert_config=expert,
            balance_config=balance,
            num_layers=12,
        )
        assert config.num_layers == 12
        assert config.router_config.num_experts == 8

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        router = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 1.0)
        expert = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        balance = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        config = MoEConfig(router, expert, balance, 12)
        with pytest.raises(AttributeError):
            config.num_layers = 24  # type: ignore[misc]


class TestMoEStats:
    """Tests for MoEStats dataclass."""

    def test_create_stats(self) -> None:
        """Create MoE stats."""
        stats = MoEStats(
            router_entropy=2.5,
            load_balance_loss=0.05,
            expert_utilization=0.75,
            dropped_tokens=128,
        )
        assert stats.router_entropy == 2.5
        assert stats.expert_utilization == 0.75

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = MoEStats(2.5, 0.05, 0.75, 128)
        with pytest.raises(AttributeError):
            stats.dropped_tokens = 256  # type: ignore[misc]


class TestValidateRouterConfig:
    """Tests for validate_router_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 1.0)
        validate_router_config(config)

    def test_zero_num_experts_raises(self) -> None:
        """Zero num_experts raises ValueError."""
        config = RouterConfig(RouterType.TOP_K, 0, 2, 0.1, 1.0)
        with pytest.raises(ValueError, match="num_experts must be positive"):
            validate_router_config(config)

    def test_negative_num_experts_raises(self) -> None:
        """Negative num_experts raises ValueError."""
        config = RouterConfig(RouterType.TOP_K, -1, 2, 0.1, 1.0)
        with pytest.raises(ValueError, match="num_experts must be positive"):
            validate_router_config(config)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        config = RouterConfig(RouterType.TOP_K, 8, 0, 0.1, 1.0)
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_router_config(config)

    def test_top_k_exceeds_num_experts_raises(self) -> None:
        """top_k > num_experts raises ValueError."""
        config = RouterConfig(RouterType.TOP_K, 8, 10, 0.1, 1.0)
        with pytest.raises(ValueError, match=r"top_k.*cannot exceed"):
            validate_router_config(config)

    def test_negative_jitter_noise_raises(self) -> None:
        """Negative jitter_noise raises ValueError."""
        config = RouterConfig(RouterType.TOP_K, 8, 2, -0.1, 1.0)
        with pytest.raises(ValueError, match="jitter_noise must be non-negative"):
            validate_router_config(config)

    def test_zero_temperature_raises(self) -> None:
        """Zero temperature raises ValueError."""
        config = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 0.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_router_config(config)

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature raises ValueError."""
        config = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, -1.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_router_config(config)


class TestValidateLoadBalanceConfig:
    """Tests for validate_load_balance_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        validate_load_balance_config(config)

    def test_negative_loss_weight_raises(self) -> None:
        """Negative loss_weight raises ValueError."""
        config = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, -0.01, 1.25, False)
        with pytest.raises(ValueError, match="loss_weight must be non-negative"):
            validate_load_balance_config(config)

    def test_zero_capacity_factor_raises(self) -> None:
        """Zero capacity_factor raises ValueError."""
        config = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 0.0, False)
        with pytest.raises(ValueError, match="capacity_factor must be positive"):
            validate_load_balance_config(config)

    def test_negative_capacity_factor_raises(self) -> None:
        """Negative capacity_factor raises ValueError."""
        config = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, -1.0, False)
        with pytest.raises(ValueError, match="capacity_factor must be positive"):
            validate_load_balance_config(config)


class TestValidateExpertConfig:
    """Tests for validate_expert_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        validate_expert_config(config)

    def test_zero_hidden_dim_raises(self) -> None:
        """Zero hidden_dim raises ValueError."""
        config = ExpertConfig(0, ExpertActivation.GELU, 0.1, False)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_expert_config(config)

    def test_negative_hidden_dim_raises(self) -> None:
        """Negative hidden_dim raises ValueError."""
        config = ExpertConfig(-1, ExpertActivation.GELU, 0.1, False)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            validate_expert_config(config)

    def test_negative_dropout_raises(self) -> None:
        """Negative dropout raises ValueError."""
        config = ExpertConfig(4096, ExpertActivation.GELU, -0.1, False)
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            validate_expert_config(config)

    def test_dropout_over_one_raises(self) -> None:
        """Dropout > 1 raises ValueError."""
        config = ExpertConfig(4096, ExpertActivation.GELU, 1.5, False)
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            validate_expert_config(config)


class TestValidateMoEConfig:
    """Tests for validate_moe_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        router = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 1.0)
        expert = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        balance = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        config = MoEConfig(router, expert, balance, 12)
        validate_moe_config(config)

    def test_zero_num_layers_raises(self) -> None:
        """Zero num_layers raises ValueError."""
        router = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 1.0)
        expert = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        balance = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        config = MoEConfig(router, expert, balance, 0)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_moe_config(config)

    def test_negative_num_layers_raises(self) -> None:
        """Negative num_layers raises ValueError."""
        router = RouterConfig(RouterType.TOP_K, 8, 2, 0.1, 1.0)
        expert = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        balance = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        config = MoEConfig(router, expert, balance, -1)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_moe_config(config)

    def test_invalid_router_config_raises(self) -> None:
        """Invalid router config raises ValueError."""
        router = RouterConfig(RouterType.TOP_K, 0, 2, 0.1, 1.0)  # Invalid
        expert = ExpertConfig(4096, ExpertActivation.GELU, 0.1, False)
        balance = LoadBalanceConfig(LoadBalancingLoss.AUXILIARY, 0.01, 1.25, False)
        config = MoEConfig(router, expert, balance, 12)
        with pytest.raises(ValueError, match="num_experts must be positive"):
            validate_moe_config(config)


class TestCreateRouterConfig:
    """Tests for create_router_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_router_config()
        assert config.router_type == RouterType.TOP_K
        assert config.num_experts == 8
        assert config.top_k == 2

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_router_config(
            router_type="expert_choice",
            num_experts=16,
            top_k=4,
        )
        assert config.router_type == RouterType.EXPERT_CHOICE
        assert config.num_experts == 16
        assert config.top_k == 4

    def test_with_enum_router_type(self) -> None:
        """Create with enum router type."""
        config = create_router_config(router_type=RouterType.SOFT)
        assert config.router_type == RouterType.SOFT

    @pytest.mark.parametrize(
        "router_type",
        ["top_k", "expert_choice", "soft", "hash"],
    )
    def test_all_router_types(self, router_type: str) -> None:
        """Test all router types."""
        config = create_router_config(router_type=router_type)
        assert config.router_type.value == router_type

    def test_invalid_num_experts_raises(self) -> None:
        """Invalid num_experts raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be positive"):
            create_router_config(num_experts=0)


class TestCreateLoadBalanceConfig:
    """Tests for create_load_balance_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_load_balance_config()
        assert config.loss_type == LoadBalancingLoss.AUXILIARY
        assert config.loss_weight == 0.01
        assert config.capacity_factor == 1.25

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_load_balance_config(
            loss_type="z_loss",
            loss_weight=0.001,
            capacity_factor=1.5,
            drop_tokens=True,
        )
        assert config.loss_type == LoadBalancingLoss.Z_LOSS
        assert config.loss_weight == 0.001
        assert config.drop_tokens is True

    def test_with_enum_loss_type(self) -> None:
        """Create with enum loss type."""
        config = create_load_balance_config(loss_type=LoadBalancingLoss.SWITCH)
        assert config.loss_type == LoadBalancingLoss.SWITCH

    @pytest.mark.parametrize(
        "loss_type",
        ["auxiliary", "z_loss", "switch", "none"],
    )
    def test_all_loss_types(self, loss_type: str) -> None:
        """Test all loss types."""
        config = create_load_balance_config(loss_type=loss_type)
        assert config.loss_type.value == loss_type

    def test_invalid_loss_weight_raises(self) -> None:
        """Invalid loss_weight raises ValueError."""
        with pytest.raises(ValueError, match="loss_weight must be non-negative"):
            create_load_balance_config(loss_weight=-0.1)


class TestCreateExpertConfig:
    """Tests for create_expert_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_expert_config()
        assert config.hidden_dim == 4096
        assert config.activation == ExpertActivation.GELU
        assert config.dropout == 0.1

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_expert_config(
            hidden_dim=8192,
            activation="swiglu",
            dropout=0.2,
            shared_expert=True,
        )
        assert config.hidden_dim == 8192
        assert config.activation == ExpertActivation.SWIGLU
        assert config.shared_expert is True

    def test_with_enum_activation(self) -> None:
        """Create with enum activation."""
        config = create_expert_config(activation=ExpertActivation.GEGLU)
        assert config.activation == ExpertActivation.GEGLU

    @pytest.mark.parametrize(
        "activation",
        ["relu", "gelu", "swiglu", "geglu"],
    )
    def test_all_activations(self, activation: str) -> None:
        """Test all activation types."""
        config = create_expert_config(activation=activation)
        assert config.activation.value == activation

    def test_invalid_hidden_dim_raises(self) -> None:
        """Invalid hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            create_expert_config(hidden_dim=0)


class TestCreateMoEConfig:
    """Tests for create_moe_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_moe_config()
        assert config.router_config.num_experts == 8
        assert config.expert_config.hidden_dim == 4096
        assert config.num_layers == 12

    def test_custom_config(self) -> None:
        """Create custom config."""
        router = create_router_config(num_experts=16, top_k=4)
        config = create_moe_config(router_config=router, num_layers=24)
        assert config.router_config.num_experts == 16
        assert config.num_layers == 24

    def test_all_custom_configs(self) -> None:
        """Create with all custom configs."""
        router = create_router_config(num_experts=32)
        expert = create_expert_config(hidden_dim=16384)
        balance = create_load_balance_config(loss_weight=0.02)
        config = create_moe_config(
            router_config=router,
            expert_config=expert,
            balance_config=balance,
            num_layers=32,
        )
        assert config.router_config.num_experts == 32
        assert config.expert_config.hidden_dim == 16384
        assert config.balance_config.loss_weight == 0.02

    def test_invalid_num_layers_raises(self) -> None:
        """Invalid num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            create_moe_config(num_layers=0)


class TestCreateMoEStats:
    """Tests for create_moe_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_moe_stats()
        assert stats.router_entropy == 0.0
        assert stats.dropped_tokens == 0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_moe_stats(
            router_entropy=2.5,
            load_balance_loss=0.05,
            expert_utilization=0.75,
            dropped_tokens=128,
        )
        assert stats.router_entropy == 2.5
        assert stats.expert_utilization == 0.75

    def test_negative_router_entropy_raises(self) -> None:
        """Negative router_entropy raises ValueError."""
        with pytest.raises(ValueError, match="router_entropy must be non-negative"):
            create_moe_stats(router_entropy=-1.0)

    def test_negative_load_balance_loss_raises(self) -> None:
        """Negative load_balance_loss raises ValueError."""
        with pytest.raises(ValueError, match="load_balance_loss must be non-negative"):
            create_moe_stats(load_balance_loss=-0.1)

    def test_expert_utilization_over_one_raises(self) -> None:
        """expert_utilization > 1 raises ValueError."""
        with pytest.raises(ValueError, match="expert_utilization must be between"):
            create_moe_stats(expert_utilization=1.5)

    def test_negative_expert_utilization_raises(self) -> None:
        """Negative expert_utilization raises ValueError."""
        with pytest.raises(ValueError, match="expert_utilization must be between"):
            create_moe_stats(expert_utilization=-0.1)

    def test_negative_dropped_tokens_raises(self) -> None:
        """Negative dropped_tokens raises ValueError."""
        with pytest.raises(ValueError, match="dropped_tokens must be non-negative"):
            create_moe_stats(dropped_tokens=-1)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_router_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_router_types()
        assert types == sorted(types)
        assert "top_k" in types

    def test_list_router_types_complete(self) -> None:
        """Returns all router types."""
        types = list_router_types()
        assert len(types) == 4

    def test_list_load_balancing_losses_sorted(self) -> None:
        """Returns sorted list."""
        losses = list_load_balancing_losses()
        assert losses == sorted(losses)
        assert "auxiliary" in losses

    def test_list_load_balancing_losses_complete(self) -> None:
        """Returns all loss types."""
        losses = list_load_balancing_losses()
        assert len(losses) == 4

    def test_list_expert_activations_sorted(self) -> None:
        """Returns sorted list."""
        activations = list_expert_activations()
        assert activations == sorted(activations)
        assert "gelu" in activations

    def test_list_expert_activations_complete(self) -> None:
        """Returns all activations."""
        activations = list_expert_activations()
        assert len(activations) == 4


class TestGetRouterType:
    """Tests for get_router_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("top_k", RouterType.TOP_K),
            ("expert_choice", RouterType.EXPERT_CHOICE),
            ("soft", RouterType.SOFT),
            ("hash", RouterType.HASH),
        ],
    )
    def test_all_types(self, name: str, expected: RouterType) -> None:
        """Test all valid router types."""
        assert get_router_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="router_type must be one of"):
            get_router_type("invalid")


class TestGetLoadBalancingLoss:
    """Tests for get_load_balancing_loss function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("auxiliary", LoadBalancingLoss.AUXILIARY),
            ("z_loss", LoadBalancingLoss.Z_LOSS),
            ("switch", LoadBalancingLoss.SWITCH),
            ("none", LoadBalancingLoss.NONE),
        ],
    )
    def test_all_losses(self, name: str, expected: LoadBalancingLoss) -> None:
        """Test all valid load balancing losses."""
        assert get_load_balancing_loss(name) == expected

    def test_invalid_loss_raises(self) -> None:
        """Invalid loss raises ValueError."""
        with pytest.raises(ValueError, match="load_balancing_loss must be one of"):
            get_load_balancing_loss("invalid")


class TestGetExpertActivation:
    """Tests for get_expert_activation function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("relu", ExpertActivation.RELU),
            ("gelu", ExpertActivation.GELU),
            ("swiglu", ExpertActivation.SWIGLU),
            ("geglu", ExpertActivation.GEGLU),
        ],
    )
    def test_all_activations(self, name: str, expected: ExpertActivation) -> None:
        """Test all valid activations."""
        assert get_expert_activation(name) == expected

    def test_invalid_activation_raises(self) -> None:
        """Invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="expert_activation must be one of"):
            get_expert_activation("invalid")


class TestCalculateCapacityFactor:
    """Tests for calculate_capacity_factor function."""

    def test_basic_calculation(self) -> None:
        """Calculate basic capacity factor."""
        factor = calculate_capacity_factor(1024, 8, top_k=2)
        assert 1.0 < factor < 2.0

    def test_larger_batch(self) -> None:
        """Larger batch maintains reasonable factor."""
        factor = calculate_capacity_factor(2048, 16, top_k=2)
        assert factor > 1.0

    def test_default_buffer_ratio(self) -> None:
        """Default buffer ratio gives expected factor."""
        factor = calculate_capacity_factor(1024, 8)
        assert factor == pytest.approx(1.25)

    def test_custom_buffer_ratio(self) -> None:
        """Custom buffer ratio affects factor."""
        factor = calculate_capacity_factor(1024, 8, buffer_ratio=0.5)
        assert factor == pytest.approx(1.5)

    def test_small_batch_adjustment(self) -> None:
        """Small batches get adjusted factor."""
        factor = calculate_capacity_factor(4, 8, top_k=1)
        assert factor >= 2.0

    def test_zero_num_tokens_raises(self) -> None:
        """Zero num_tokens raises ValueError."""
        with pytest.raises(ValueError, match="num_tokens must be positive"):
            calculate_capacity_factor(0, 8)

    def test_zero_num_experts_raises(self) -> None:
        """Zero num_experts raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be positive"):
            calculate_capacity_factor(1024, 0)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            calculate_capacity_factor(1024, 8, top_k=0)

    def test_top_k_exceeds_num_experts_raises(self) -> None:
        """top_k > num_experts raises ValueError."""
        with pytest.raises(ValueError, match=r"top_k.*cannot exceed"):
            calculate_capacity_factor(1024, 8, top_k=10)

    def test_negative_buffer_ratio_raises(self) -> None:
        """Negative buffer_ratio raises ValueError."""
        with pytest.raises(ValueError, match="buffer_ratio must be non-negative"):
            calculate_capacity_factor(1024, 8, buffer_ratio=-0.1)


class TestEstimateExpertUtilization:
    """Tests for estimate_expert_utilization function."""

    def test_half_utilization(self) -> None:
        """Half the experts used."""
        probs = (0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0)
        utilization = estimate_expert_utilization(probs)
        assert utilization == 0.5

    def test_full_utilization(self) -> None:
        """All experts used."""
        probs = (0.125,) * 8
        utilization = estimate_expert_utilization(probs)
        assert utilization == 1.0

    def test_no_utilization(self) -> None:
        """No experts above threshold."""
        probs = (0.005,) * 8
        utilization = estimate_expert_utilization(probs)
        assert utilization == 0.0

    def test_custom_threshold(self) -> None:
        """Custom threshold affects utilization."""
        probs = (0.15, 0.15, 0.05, 0.05, 0.2, 0.2, 0.15, 0.05)
        utilization = estimate_expert_utilization(probs, threshold=0.1)
        # probs > 0.1 are: 0.15, 0.15, 0.2, 0.2, 0.15 = 5 out of 8 = 0.625
        assert utilization == 0.625

    def test_empty_probs_raises(self) -> None:
        """Empty probs raises ValueError."""
        with pytest.raises(ValueError, match="router_probs cannot be empty"):
            estimate_expert_utilization(())

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            estimate_expert_utilization((0.5, 0.5), threshold=-0.1)


class TestCalculateRouterEntropy:
    """Tests for calculate_router_entropy function."""

    def test_uniform_two_experts(self) -> None:
        """Uniform distribution over 2 experts gives ln(2)."""
        probs = (0.5, 0.5)
        entropy = calculate_router_entropy(probs)
        assert entropy == pytest.approx(math.log(2), abs=0.01)

    def test_uniform_four_experts(self) -> None:
        """Uniform distribution over 4 experts gives ln(4)."""
        probs = (0.25, 0.25, 0.25, 0.25)
        entropy = calculate_router_entropy(probs)
        assert entropy == pytest.approx(math.log(4), abs=0.01)

    def test_deterministic_routing(self) -> None:
        """Deterministic routing gives zero entropy."""
        probs = (1.0, 0.0, 0.0, 0.0)
        entropy = calculate_router_entropy(probs)
        assert entropy == pytest.approx(0.0, abs=0.001)

    def test_empty_probs_raises(self) -> None:
        """Empty probs raises ValueError."""
        with pytest.raises(ValueError, match="router_probs cannot be empty"):
            calculate_router_entropy(())

    def test_invalid_prob_sum_raises(self) -> None:
        """Probabilities not summing to 1 raises ValueError."""
        with pytest.raises(ValueError, match="router_probs must sum to 1"):
            calculate_router_entropy((0.5, 0.25))


class TestCalculateLoadBalanceLoss:
    """Tests for calculate_load_balance_loss function."""

    def test_balanced_case(self) -> None:
        """Perfectly balanced case."""
        probs = (0.25, 0.25, 0.25, 0.25)
        counts = (100, 100, 100, 100)
        loss = calculate_load_balance_loss(probs, counts)
        # num_experts * sum(prob_i * count_frac_i) = 4 * (0.25*0.25 * 4) = 1.0
        assert loss == pytest.approx(1.0)

    def test_imbalanced_case(self) -> None:
        """Imbalanced case has higher loss."""
        probs = (0.5, 0.5, 0.0, 0.0)
        counts = (200, 200, 0, 0)
        loss = calculate_load_balance_loss(probs, counts)
        # Imbalanced: 4 * (0.5*0.5 + 0.5*0.5 + 0 + 0) = 4 * 0.5 = 2.0
        assert loss == pytest.approx(2.0)

    def test_none_loss_type(self) -> None:
        """None loss type returns zero."""
        probs = (0.25, 0.25, 0.25, 0.25)
        counts = (100, 100, 100, 100)
        loss = calculate_load_balance_loss(probs, counts, LoadBalancingLoss.NONE)
        assert loss == 0.0

    def test_z_loss_type(self) -> None:
        """Z-loss type computes variance-based loss."""
        probs = (0.25, 0.25, 0.25, 0.25)
        counts = (100, 100, 100, 100)
        loss = calculate_load_balance_loss(probs, counts, LoadBalancingLoss.Z_LOSS)
        # Uniform probs have zero variance
        assert loss == pytest.approx(0.0, abs=0.001)

    def test_switch_loss_type(self) -> None:
        """Switch loss type computes similar to auxiliary."""
        probs = (0.25, 0.25, 0.25, 0.25)
        counts = (100, 100, 100, 100)
        loss = calculate_load_balance_loss(probs, counts, LoadBalancingLoss.SWITCH)
        # Same as auxiliary: num_experts * sum(prob_i * count_frac_i) = 1.0
        assert loss == pytest.approx(1.0)

    def test_zero_total_tokens(self) -> None:
        """Zero total tokens returns zero loss."""
        probs = (0.25, 0.25, 0.25, 0.25)
        counts = (0, 0, 0, 0)
        loss = calculate_load_balance_loss(probs, counts)
        assert loss == 0.0

    def test_empty_probs_raises(self) -> None:
        """Empty probs raises ValueError."""
        with pytest.raises(ValueError, match="router_probs cannot be empty"):
            calculate_load_balance_loss((), (100,))

    def test_empty_counts_raises(self) -> None:
        """Empty counts raises ValueError."""
        with pytest.raises(ValueError, match="expert_counts cannot be empty"):
            calculate_load_balance_loss((0.5, 0.5), ())

    def test_length_mismatch_raises(self) -> None:
        """Length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_load_balance_loss((0.5, 0.5), (100, 100, 100))


class TestFormatMoEStats:
    """Tests for format_moe_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_moe_stats(
            router_entropy=2.5,
            load_balance_loss=0.05,
            expert_utilization=0.75,
            dropped_tokens=128,
        )
        formatted = format_moe_stats(stats)
        assert "Router Entropy: 2.50" in formatted
        assert "Load Balance Loss: 0.0500" in formatted
        assert "Expert Utilization: 75.0%" in formatted
        assert "Dropped Tokens: 128" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_moe_stats()
        formatted = format_moe_stats(stats)
        assert "Router Entropy:" in formatted
        assert "Load Balance Loss:" in formatted
        assert "Expert Utilization:" in formatted
        assert "Dropped Tokens:" in formatted

    def test_contains_header(self) -> None:
        """Formatted string has header."""
        stats = create_moe_stats()
        formatted = format_moe_stats(stats)
        assert "MoE Stats:" in formatted


class TestGetRecommendedMoEConfig:
    """Tests for get_recommended_moe_config function."""

    def test_small_model(self) -> None:
        """Config for small model (< 1B params)."""
        config = get_recommended_moe_config(500_000_000)
        assert config.router_config.num_experts == 4
        assert config.router_config.top_k == 1
        assert config.expert_config.shared_expert is False

    def test_medium_model(self) -> None:
        """Config for medium model (1B - 10B params)."""
        config = get_recommended_moe_config(7_000_000_000)
        assert config.router_config.num_experts == 8
        assert config.router_config.top_k == 2
        assert config.expert_config.hidden_dim == 4096

    def test_large_model(self) -> None:
        """Config for large model (10B - 100B params)."""
        config = get_recommended_moe_config(50_000_000_000)
        assert config.router_config.num_experts == 16
        assert config.router_config.top_k == 2
        assert config.expert_config.shared_expert is True

    def test_huge_model(self) -> None:
        """Config for huge model (100B+ params)."""
        config = get_recommended_moe_config(200_000_000_000)
        assert config.router_config.num_experts == 64
        assert config.router_config.top_k == 4
        assert config.expert_config.hidden_dim == 16384

    def test_all_configs_valid(self) -> None:
        """All recommended configs are valid."""
        for params in [100_000_000, 5_000_000_000, 50_000_000_000, 500_000_000_000]:
            config = get_recommended_moe_config(params)
            validate_moe_config(config)

    def test_zero_params_raises(self) -> None:
        """Zero model_params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            get_recommended_moe_config(0)

    def test_negative_params_raises(self) -> None:
        """Negative model_params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            get_recommended_moe_config(-1)

    def test_router_type_is_top_k(self) -> None:
        """All configs use TOP_K router."""
        for params in [100_000_000, 5_000_000_000, 50_000_000_000]:
            config = get_recommended_moe_config(params)
            assert config.router_config.router_type == RouterType.TOP_K

    def test_activation_is_swiglu(self) -> None:
        """All configs use SWIGLU activation."""
        for params in [100_000_000, 5_000_000_000, 50_000_000_000]:
            config = get_recommended_moe_config(params)
            assert config.expert_config.activation == ExpertActivation.SWIGLU
