"""Tests for training.federated module."""

from __future__ import annotations

import pytest

from hf_gtc.training.federated import (
    VALID_AGGREGATION_METHODS,
    VALID_CLIENT_SELECTION_STRATEGIES,
    VALID_PRIVACY_MECHANISMS,
    AggregationMethod,
    ClientConfig,
    ClientSelectionStrategy,
    FederatedConfig,
    FederatedStats,
    PrivacyConfig,
    PrivacyMechanism,
    aggregate_updates,
    calculate_client_weights,
    calculate_privacy_budget,
    create_client_config,
    create_federated_config,
    create_federated_stats,
    create_privacy_config,
    estimate_convergence_rounds,
    format_federated_stats,
    get_aggregation_method,
    get_client_selection_strategy,
    get_privacy_mechanism,
    get_recommended_federated_config,
    list_aggregation_methods,
    list_client_selection_strategies,
    list_privacy_mechanisms,
    select_clients,
    validate_client_config,
    validate_federated_config,
    validate_federated_stats,
    validate_privacy_config,
)


class TestAggregationMethod:
    """Tests for AggregationMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in AggregationMethod:
            assert isinstance(method.value, str)

    def test_fedavg_value(self) -> None:
        """FedAvg has correct value."""
        assert AggregationMethod.FEDAVG.value == "fedavg"

    def test_fedprox_value(self) -> None:
        """FedProx has correct value."""
        assert AggregationMethod.FEDPROX.value == "fedprox"

    def test_scaffold_value(self) -> None:
        """SCAFFOLD has correct value."""
        assert AggregationMethod.SCAFFOLD.value == "scaffold"

    def test_fedadam_value(self) -> None:
        """FedAdam has correct value."""
        assert AggregationMethod.FEDADAM.value == "fedadam"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_AGGREGATION_METHODS is a frozenset."""
        assert isinstance(VALID_AGGREGATION_METHODS, frozenset)


class TestClientSelectionStrategy:
    """Tests for ClientSelectionStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ClientSelectionStrategy:
            assert isinstance(strategy.value, str)

    def test_random_value(self) -> None:
        """Random has correct value."""
        assert ClientSelectionStrategy.RANDOM.value == "random"

    def test_round_robin_value(self) -> None:
        """Round robin has correct value."""
        assert ClientSelectionStrategy.ROUND_ROBIN.value == "round_robin"

    def test_resource_based_value(self) -> None:
        """Resource based has correct value."""
        assert ClientSelectionStrategy.RESOURCE_BASED.value == "resource_based"

    def test_contribution_value(self) -> None:
        """Contribution has correct value."""
        assert ClientSelectionStrategy.CONTRIBUTION.value == "contribution"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_CLIENT_SELECTION_STRATEGIES is a frozenset."""
        assert isinstance(VALID_CLIENT_SELECTION_STRATEGIES, frozenset)


class TestPrivacyMechanism:
    """Tests for PrivacyMechanism enum."""

    def test_all_mechanisms_have_values(self) -> None:
        """All mechanisms have string values."""
        for mechanism in PrivacyMechanism:
            assert isinstance(mechanism.value, str)

    def test_none_value(self) -> None:
        """None has correct value."""
        assert PrivacyMechanism.NONE.value == "none"

    def test_local_dp_value(self) -> None:
        """Local DP has correct value."""
        assert PrivacyMechanism.LOCAL_DP.value == "local_dp"

    def test_central_dp_value(self) -> None:
        """Central DP has correct value."""
        assert PrivacyMechanism.CENTRAL_DP.value == "central_dp"

    def test_secure_aggregation_value(self) -> None:
        """Secure aggregation has correct value."""
        assert PrivacyMechanism.SECURE_AGGREGATION.value == "secure_aggregation"

    def test_valid_mechanisms_frozenset(self) -> None:
        """VALID_PRIVACY_MECHANISMS is a frozenset."""
        assert isinstance(VALID_PRIVACY_MECHANISMS, frozenset)


class TestFederatedConfig:
    """Tests for FederatedConfig dataclass."""

    def test_create_config(self) -> None:
        """Create federated config."""
        config = FederatedConfig(
            aggregation_method=AggregationMethod.FEDAVG,
            num_clients=100,
            rounds=50,
            local_epochs=5,
            client_fraction=0.1,
        )
        assert config.num_clients == 100
        assert config.rounds == 50
        assert config.local_epochs == 5
        assert config.client_fraction == 0.1

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FederatedConfig(
            aggregation_method=AggregationMethod.FEDAVG,
            num_clients=100,
            rounds=50,
            local_epochs=5,
            client_fraction=0.1,
        )
        with pytest.raises(AttributeError):
            config.num_clients = 200  # type: ignore[misc]


class TestPrivacyConfig:
    """Tests for PrivacyConfig dataclass."""

    def test_create_config(self) -> None:
        """Create privacy config."""
        config = PrivacyConfig(
            mechanism=PrivacyMechanism.LOCAL_DP,
            epsilon=1.0,
            delta=1e-5,
            clip_norm=1.0,
            noise_multiplier=1.1,
        )
        assert config.mechanism == PrivacyMechanism.LOCAL_DP
        assert config.epsilon == 1.0
        assert config.delta == 1e-5

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PrivacyConfig(
            mechanism=PrivacyMechanism.NONE,
            epsilon=1.0,
            delta=1e-5,
            clip_norm=1.0,
            noise_multiplier=1.1,
        )
        with pytest.raises(AttributeError):
            config.epsilon = 2.0  # type: ignore[misc]


class TestClientConfig:
    """Tests for ClientConfig dataclass."""

    def test_create_config(self) -> None:
        """Create client config."""
        config = ClientConfig(
            client_id="client_001",
            data_size=1000,
            compute_capacity=0.8,
            availability=0.95,
        )
        assert config.client_id == "client_001"
        assert config.data_size == 1000
        assert config.compute_capacity == 0.8
        assert config.availability == 0.95

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ClientConfig(
            client_id="client_001",
            data_size=1000,
            compute_capacity=0.8,
            availability=0.95,
        )
        with pytest.raises(AttributeError):
            config.data_size = 2000  # type: ignore[misc]


class TestFederatedStats:
    """Tests for FederatedStats dataclass."""

    def test_create_stats(self) -> None:
        """Create federated stats."""
        stats = FederatedStats(
            global_round=10,
            participating_clients=20,
            aggregation_time=5.5,
            privacy_budget_spent=0.5,
        )
        assert stats.global_round == 10
        assert stats.participating_clients == 20
        assert stats.aggregation_time == 5.5
        assert stats.privacy_budget_spent == 0.5

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = FederatedStats(
            global_round=10,
            participating_clients=20,
            aggregation_time=5.5,
            privacy_budget_spent=0.5,
        )
        with pytest.raises(AttributeError):
            stats.global_round = 20  # type: ignore[misc]


class TestValidateFederatedConfig:
    """Tests for validate_federated_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FederatedConfig(AggregationMethod.FEDAVG, 100, 50, 5, 0.1)
        validate_federated_config(config)

    def test_zero_num_clients_raises(self) -> None:
        """Zero num_clients raises ValueError."""
        config = FederatedConfig(AggregationMethod.FEDAVG, 0, 50, 5, 0.1)
        with pytest.raises(ValueError, match="num_clients must be positive"):
            validate_federated_config(config)

    def test_zero_rounds_raises(self) -> None:
        """Zero rounds raises ValueError."""
        config = FederatedConfig(AggregationMethod.FEDAVG, 100, 0, 5, 0.1)
        with pytest.raises(ValueError, match="rounds must be positive"):
            validate_federated_config(config)

    def test_zero_local_epochs_raises(self) -> None:
        """Zero local_epochs raises ValueError."""
        config = FederatedConfig(AggregationMethod.FEDAVG, 100, 50, 0, 0.1)
        with pytest.raises(ValueError, match="local_epochs must be positive"):
            validate_federated_config(config)

    def test_invalid_client_fraction_raises(self) -> None:
        """Invalid client_fraction raises ValueError."""
        config = FederatedConfig(AggregationMethod.FEDAVG, 100, 50, 5, 0.0)
        with pytest.raises(ValueError, match="client_fraction must be in"):
            validate_federated_config(config)

    def test_client_fraction_over_one_raises(self) -> None:
        """Client fraction over 1 raises ValueError."""
        config = FederatedConfig(AggregationMethod.FEDAVG, 100, 50, 5, 1.5)
        with pytest.raises(ValueError, match="client_fraction must be in"):
            validate_federated_config(config)


class TestValidatePrivacyConfig:
    """Tests for validate_privacy_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PrivacyConfig(PrivacyMechanism.LOCAL_DP, 1.0, 1e-5, 1.0, 1.1)
        validate_privacy_config(config)

    def test_none_mechanism_skips_validation(self) -> None:
        """None mechanism skips other validations."""
        config = PrivacyConfig(PrivacyMechanism.NONE, -1.0, 2.0, -1.0, -1.0)
        validate_privacy_config(config)  # Should not raise

    def test_negative_epsilon_raises(self) -> None:
        """Negative epsilon raises ValueError."""
        config = PrivacyConfig(PrivacyMechanism.LOCAL_DP, -1.0, 1e-5, 1.0, 1.1)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            validate_privacy_config(config)

    def test_invalid_delta_raises(self) -> None:
        """Invalid delta raises ValueError."""
        config = PrivacyConfig(PrivacyMechanism.LOCAL_DP, 1.0, 1.5, 1.0, 1.1)
        with pytest.raises(ValueError, match="delta must be in"):
            validate_privacy_config(config)

    def test_zero_delta_raises(self) -> None:
        """Zero delta raises ValueError."""
        config = PrivacyConfig(PrivacyMechanism.LOCAL_DP, 1.0, 0.0, 1.0, 1.1)
        with pytest.raises(ValueError, match="delta must be in"):
            validate_privacy_config(config)

    def test_negative_clip_norm_raises(self) -> None:
        """Negative clip_norm raises ValueError."""
        config = PrivacyConfig(PrivacyMechanism.LOCAL_DP, 1.0, 1e-5, -1.0, 1.1)
        with pytest.raises(ValueError, match="clip_norm must be positive"):
            validate_privacy_config(config)

    def test_negative_noise_multiplier_raises(self) -> None:
        """Negative noise_multiplier raises ValueError."""
        config = PrivacyConfig(PrivacyMechanism.LOCAL_DP, 1.0, 1e-5, 1.0, -1.1)
        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            validate_privacy_config(config)


class TestValidateClientConfig:
    """Tests for validate_client_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ClientConfig("client_001", 1000, 0.8, 0.95)
        validate_client_config(config)

    def test_empty_client_id_raises(self) -> None:
        """Empty client_id raises ValueError."""
        config = ClientConfig("", 1000, 0.8, 0.95)
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            validate_client_config(config)

    def test_whitespace_client_id_raises(self) -> None:
        """Whitespace client_id raises ValueError."""
        config = ClientConfig("   ", 1000, 0.8, 0.95)
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            validate_client_config(config)

    def test_zero_data_size_raises(self) -> None:
        """Zero data_size raises ValueError."""
        config = ClientConfig("client_001", 0, 0.8, 0.95)
        with pytest.raises(ValueError, match="data_size must be positive"):
            validate_client_config(config)

    def test_invalid_compute_capacity_raises(self) -> None:
        """Invalid compute_capacity raises ValueError."""
        config = ClientConfig("client_001", 1000, 0.0, 0.95)
        with pytest.raises(ValueError, match="compute_capacity must be in"):
            validate_client_config(config)

    def test_compute_capacity_over_one_raises(self) -> None:
        """Compute capacity over 1 raises ValueError."""
        config = ClientConfig("client_001", 1000, 1.5, 0.95)
        with pytest.raises(ValueError, match="compute_capacity must be in"):
            validate_client_config(config)

    def test_invalid_availability_raises(self) -> None:
        """Invalid availability raises ValueError."""
        config = ClientConfig("client_001", 1000, 0.8, 0.0)
        with pytest.raises(ValueError, match="availability must be in"):
            validate_client_config(config)


class TestValidateFederatedStats:
    """Tests for validate_federated_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = FederatedStats(10, 20, 5.5, 0.5)
        validate_federated_stats(stats)

    def test_negative_global_round_raises(self) -> None:
        """Negative global_round raises ValueError."""
        stats = FederatedStats(-1, 20, 5.5, 0.5)
        with pytest.raises(ValueError, match="global_round must be non-negative"):
            validate_federated_stats(stats)

    def test_negative_participating_clients_raises(self) -> None:
        """Negative participating_clients raises ValueError."""
        stats = FederatedStats(10, -1, 5.5, 0.5)
        with pytest.raises(
            ValueError, match="participating_clients must be non-negative"
        ):
            validate_federated_stats(stats)

    def test_negative_aggregation_time_raises(self) -> None:
        """Negative aggregation_time raises ValueError."""
        stats = FederatedStats(10, 20, -1.0, 0.5)
        with pytest.raises(ValueError, match="aggregation_time must be non-negative"):
            validate_federated_stats(stats)

    def test_negative_privacy_budget_raises(self) -> None:
        """Negative privacy_budget_spent raises ValueError."""
        stats = FederatedStats(10, 20, 5.5, -0.5)
        with pytest.raises(
            ValueError, match="privacy_budget_spent must be non-negative"
        ):
            validate_federated_stats(stats)


class TestCreateFederatedConfig:
    """Tests for create_federated_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_federated_config()
        assert config.aggregation_method == AggregationMethod.FEDAVG
        assert config.num_clients == 10
        assert config.rounds == 10

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_federated_config(
            aggregation_method="fedprox",
            num_clients=100,
            client_fraction=0.2,
        )
        assert config.aggregation_method == AggregationMethod.FEDPROX
        assert config.num_clients == 100
        assert config.client_fraction == 0.2

    def test_invalid_num_clients_raises(self) -> None:
        """Invalid num_clients raises ValueError."""
        with pytest.raises(ValueError, match="num_clients must be positive"):
            create_federated_config(num_clients=0)

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="aggregation_method must be one of"):
            create_federated_config(aggregation_method="invalid")


class TestCreatePrivacyConfig:
    """Tests for create_privacy_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_privacy_config()
        assert config.mechanism == PrivacyMechanism.NONE

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_privacy_config(
            mechanism="local_dp",
            epsilon=0.5,
            noise_multiplier=1.5,
        )
        assert config.mechanism == PrivacyMechanism.LOCAL_DP
        assert config.epsilon == 0.5
        assert config.noise_multiplier == 1.5

    def test_invalid_epsilon_raises(self) -> None:
        """Invalid epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            create_privacy_config(mechanism="local_dp", epsilon=-1.0)

    def test_invalid_mechanism_raises(self) -> None:
        """Invalid mechanism raises ValueError."""
        with pytest.raises(ValueError, match="privacy_mechanism must be one of"):
            create_privacy_config(mechanism="invalid")


class TestCreateClientConfig:
    """Tests for create_client_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_client_config("client_001")
        assert config.client_id == "client_001"
        assert config.data_size == 1000
        assert config.compute_capacity == 1.0
        assert config.availability == 1.0

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_client_config(
            "client_002",
            data_size=5000,
            compute_capacity=0.5,
        )
        assert config.client_id == "client_002"
        assert config.data_size == 5000
        assert config.compute_capacity == 0.5

    def test_empty_client_id_raises(self) -> None:
        """Empty client_id raises ValueError."""
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            create_client_config("")


class TestCreateFederatedStats:
    """Tests for create_federated_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_federated_stats()
        assert stats.global_round == 0
        assert stats.participating_clients == 0
        assert stats.aggregation_time == 0.0
        assert stats.privacy_budget_spent == 0.0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_federated_stats(
            global_round=5,
            participating_clients=10,
            aggregation_time=2.5,
        )
        assert stats.global_round == 5
        assert stats.participating_clients == 10
        assert stats.aggregation_time == 2.5

    def test_negative_round_raises(self) -> None:
        """Negative round raises ValueError."""
        with pytest.raises(ValueError, match="global_round must be non-negative"):
            create_federated_stats(global_round=-1)


class TestListAggregationMethods:
    """Tests for list_aggregation_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        methods = list_aggregation_methods()
        assert methods == sorted(methods)

    def test_contains_fedavg(self) -> None:
        """Contains fedavg."""
        methods = list_aggregation_methods()
        assert "fedavg" in methods

    def test_contains_fedprox(self) -> None:
        """Contains fedprox."""
        methods = list_aggregation_methods()
        assert "fedprox" in methods


class TestListClientSelectionStrategies:
    """Tests for list_client_selection_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_client_selection_strategies()
        assert strategies == sorted(strategies)

    def test_contains_random(self) -> None:
        """Contains random."""
        strategies = list_client_selection_strategies()
        assert "random" in strategies

    def test_contains_round_robin(self) -> None:
        """Contains round_robin."""
        strategies = list_client_selection_strategies()
        assert "round_robin" in strategies


class TestListPrivacyMechanisms:
    """Tests for list_privacy_mechanisms function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        mechanisms = list_privacy_mechanisms()
        assert mechanisms == sorted(mechanisms)

    def test_contains_none(self) -> None:
        """Contains none."""
        mechanisms = list_privacy_mechanisms()
        assert "none" in mechanisms

    def test_contains_local_dp(self) -> None:
        """Contains local_dp."""
        mechanisms = list_privacy_mechanisms()
        assert "local_dp" in mechanisms


class TestGetAggregationMethod:
    """Tests for get_aggregation_method function."""

    def test_get_fedavg(self) -> None:
        """Get fedavg."""
        assert get_aggregation_method("fedavg") == AggregationMethod.FEDAVG

    def test_get_fedprox(self) -> None:
        """Get fedprox."""
        assert get_aggregation_method("fedprox") == AggregationMethod.FEDPROX

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="aggregation_method must be one of"):
            get_aggregation_method("invalid")


class TestGetClientSelectionStrategy:
    """Tests for get_client_selection_strategy function."""

    def test_get_random(self) -> None:
        """Get random."""
        assert get_client_selection_strategy("random") == ClientSelectionStrategy.RANDOM

    def test_get_round_robin(self) -> None:
        """Get round_robin."""
        strategy = get_client_selection_strategy("round_robin")
        assert strategy == ClientSelectionStrategy.ROUND_ROBIN

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(
            ValueError, match="client_selection_strategy must be one of"
        ):
            get_client_selection_strategy("invalid")


class TestGetPrivacyMechanism:
    """Tests for get_privacy_mechanism function."""

    def test_get_none(self) -> None:
        """Get none."""
        assert get_privacy_mechanism("none") == PrivacyMechanism.NONE

    def test_get_local_dp(self) -> None:
        """Get local_dp."""
        assert get_privacy_mechanism("local_dp") == PrivacyMechanism.LOCAL_DP

    def test_invalid_mechanism_raises(self) -> None:
        """Invalid mechanism raises ValueError."""
        with pytest.raises(ValueError, match="privacy_mechanism must be one of"):
            get_privacy_mechanism("invalid")


class TestCalculateClientWeights:
    """Tests for calculate_client_weights function."""

    def test_equal_data_sizes(self) -> None:
        """Equal data sizes give equal weights."""
        c1 = create_client_config("c1", data_size=100)
        c2 = create_client_config("c2", data_size=100)
        c3 = create_client_config("c3", data_size=100)
        weights = calculate_client_weights((c1, c2, c3))
        assert len(weights) == 3
        assert all(abs(w - 1 / 3) < 1e-10 for w in weights)

    def test_unequal_data_sizes(self) -> None:
        """Unequal data sizes give proportional weights."""
        c1 = create_client_config("c1", data_size=1000)
        c2 = create_client_config("c2", data_size=3000)
        weights = calculate_client_weights((c1, c2))
        assert weights == (0.25, 0.75)

    def test_weights_sum_to_one(self) -> None:
        """Weights sum to 1."""
        c1 = create_client_config("c1", data_size=100)
        c2 = create_client_config("c2", data_size=200)
        c3 = create_client_config("c3", data_size=300)
        weights = calculate_client_weights((c1, c2, c3))
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_empty_clients_raises(self) -> None:
        """Empty clients raises ValueError."""
        with pytest.raises(ValueError, match="clients cannot be empty"):
            calculate_client_weights(())


class TestSelectClients:
    """Tests for select_clients function."""

    def test_random_selection(self) -> None:
        """Random selection works."""
        c1 = create_client_config("c1")
        c2 = create_client_config("c2")
        c3 = create_client_config("c3")
        selected = select_clients((c1, c2, c3), 2, "random", seed=42)
        assert len(selected) == 2

    def test_random_selection_reproducible(self) -> None:
        """Random selection is reproducible with seed."""
        c1 = create_client_config("c1")
        c2 = create_client_config("c2")
        c3 = create_client_config("c3")
        clients = (c1, c2, c3)
        selected1 = select_clients(clients, 2, "random", seed=42)
        selected2 = select_clients(clients, 2, "random", seed=42)
        assert selected1 == selected2

    def test_round_robin_selection(self) -> None:
        """Round robin selection works."""
        c1 = create_client_config("c1")
        c2 = create_client_config("c2")
        c3 = create_client_config("c3")
        clients = (c1, c2, c3)
        selected_r0 = select_clients(clients, 2, "round_robin", round_number=0)
        selected_r1 = select_clients(clients, 2, "round_robin", round_number=1)
        assert selected_r0 == (c1, c2)
        assert selected_r1 == (c3, c1)

    def test_resource_based_selection(self) -> None:
        """Resource based selection prefers high capacity."""
        c1 = create_client_config("c1", compute_capacity=0.5)
        c2 = create_client_config("c2", compute_capacity=0.8)
        c3 = create_client_config("c3", compute_capacity=1.0)
        selected = select_clients((c1, c2, c3), 2, "resource_based")
        assert selected[0].compute_capacity >= selected[1].compute_capacity

    def test_contribution_selection(self) -> None:
        """Contribution selection works (currently same as random)."""
        c1 = create_client_config("c1")
        c2 = create_client_config("c2")
        c3 = create_client_config("c3")
        selected = select_clients((c1, c2, c3), 2, "contribution", seed=42)
        assert len(selected) == 2

    def test_selection_with_enum_strategy(self) -> None:
        """Selection works with ClientSelectionStrategy enum."""
        c1 = create_client_config("c1")
        c2 = create_client_config("c2")
        c3 = create_client_config("c3")
        selected = select_clients(
            (c1, c2, c3), 2, ClientSelectionStrategy.RANDOM, seed=42
        )
        assert len(selected) == 2

    def test_empty_clients_raises(self) -> None:
        """Empty clients raises ValueError."""
        with pytest.raises(ValueError, match="clients cannot be empty"):
            select_clients((), 2, "random")

    def test_zero_num_to_select_raises(self) -> None:
        """Zero num_to_select raises ValueError."""
        c1 = create_client_config("c1")
        with pytest.raises(ValueError, match="num_to_select must be positive"):
            select_clients((c1,), 0, "random")

    def test_too_many_to_select_raises(self) -> None:
        """Too many to select raises ValueError."""
        c1 = create_client_config("c1")
        with pytest.raises(ValueError, match=r"num_to_select .* cannot exceed"):
            select_clients((c1,), 5, "random")


class TestAggregateUpdates:
    """Tests for aggregate_updates function."""

    def test_equal_weights(self) -> None:
        """Equal weights average updates."""
        updates = ((1.0, 2.0), (3.0, 4.0))
        weights = (0.5, 0.5)
        result = aggregate_updates(updates, weights)
        assert result == (2.0, 3.0)

    def test_unequal_weights(self) -> None:
        """Unequal weights weighted average."""
        updates = ((1.0, 2.0), (2.0, 4.0))
        weights = (0.25, 0.75)
        result = aggregate_updates(updates, weights)
        assert result == (1.75, 3.5)

    def test_single_client(self) -> None:
        """Single client returns its update."""
        updates = ((1.0, 2.0, 3.0),)
        weights = (1.0,)
        result = aggregate_updates(updates, weights)
        assert result == (1.0, 2.0, 3.0)

    def test_fedprox_method(self) -> None:
        """FedProx method works (same as FedAvg for aggregation)."""
        updates = ((1.0, 2.0), (3.0, 4.0))
        weights = (0.5, 0.5)
        result = aggregate_updates(updates, weights, method="fedprox")
        assert result == (2.0, 3.0)

    def test_empty_updates_raises(self) -> None:
        """Empty updates raises ValueError."""
        with pytest.raises(ValueError, match="client_updates cannot be empty"):
            aggregate_updates((), (0.5,))

    def test_empty_weights_raises(self) -> None:
        """Empty weights raises ValueError."""
        with pytest.raises(ValueError, match="weights cannot be empty"):
            aggregate_updates(((1.0,),), ())

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match=r"number of updates .* must match"):
            aggregate_updates(((1.0,),), (0.5, 0.5))

    def test_inconsistent_update_lengths_raises(self) -> None:
        """Inconsistent update lengths raises ValueError."""
        with pytest.raises(ValueError, match="all client updates must have the same"):
            aggregate_updates(((1.0,), (1.0, 2.0)), (0.5, 0.5))


class TestCalculatePrivacyBudget:
    """Tests for calculate_privacy_budget function."""

    def test_no_privacy_returns_zero(self) -> None:
        """No privacy mechanism returns zero budget."""
        config = create_privacy_config(mechanism="none")
        budget = calculate_privacy_budget(config, 10, 0.1)
        assert budget == 0.0

    def test_local_dp_positive_budget(self) -> None:
        """Local DP returns positive budget."""
        config = create_privacy_config(
            mechanism="local_dp",
            epsilon=1.0,
            noise_multiplier=1.1,
        )
        budget = calculate_privacy_budget(config, 10, 0.1)
        assert budget > 0

    def test_more_rounds_more_budget(self) -> None:
        """More rounds use more budget."""
        config = create_privacy_config(
            mechanism="local_dp",
            epsilon=10.0,
            noise_multiplier=1.1,
        )
        budget_10 = calculate_privacy_budget(config, 10, 0.1)
        budget_100 = calculate_privacy_budget(config, 100, 0.1)
        assert budget_100 > budget_10

    def test_zero_rounds_raises(self) -> None:
        """Zero rounds raises ValueError."""
        config = create_privacy_config(mechanism="local_dp")
        with pytest.raises(ValueError, match="num_rounds must be positive"):
            calculate_privacy_budget(config, 0, 0.1)

    def test_invalid_sample_rate_raises(self) -> None:
        """Invalid sample rate raises ValueError."""
        config = create_privacy_config(mechanism="local_dp")
        with pytest.raises(ValueError, match="sample_rate must be in"):
            calculate_privacy_budget(config, 10, 0.0)


class TestEstimateConvergenceRounds:
    """Tests for estimate_convergence_rounds function."""

    def test_basic_estimate(self) -> None:
        """Basic estimate returns positive value."""
        rounds = estimate_convergence_rounds(100, 0.1)
        assert rounds > 0

    def test_heterogeneity_increases_rounds(self) -> None:
        """Higher heterogeneity increases rounds."""
        rounds_homo = estimate_convergence_rounds(100, 0.1, data_heterogeneity=0.1)
        rounds_hetero = estimate_convergence_rounds(100, 0.1, data_heterogeneity=0.9)
        assert rounds_hetero > rounds_homo

    def test_higher_accuracy_increases_rounds(self) -> None:
        """Higher target accuracy increases rounds."""
        rounds_low = estimate_convergence_rounds(100, 0.1, target_accuracy=0.7)
        rounds_high = estimate_convergence_rounds(100, 0.1, target_accuracy=0.99)
        assert rounds_high > rounds_low

    def test_zero_clients_raises(self) -> None:
        """Zero clients raises ValueError."""
        with pytest.raises(ValueError, match="num_clients must be positive"):
            estimate_convergence_rounds(0, 0.1)

    def test_invalid_client_fraction_raises(self) -> None:
        """Invalid client_fraction raises ValueError."""
        with pytest.raises(ValueError, match="client_fraction must be in"):
            estimate_convergence_rounds(100, 0.0)

    def test_invalid_heterogeneity_raises(self) -> None:
        """Invalid heterogeneity raises ValueError."""
        with pytest.raises(ValueError, match="data_heterogeneity must be in"):
            estimate_convergence_rounds(100, 0.1, data_heterogeneity=1.5)

    def test_invalid_target_accuracy_raises(self) -> None:
        """Invalid target_accuracy raises ValueError."""
        with pytest.raises(ValueError, match="target_accuracy must be in"):
            estimate_convergence_rounds(100, 0.1, target_accuracy=0.0)


class TestFormatFederatedStats:
    """Tests for format_federated_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_federated_stats(
            global_round=10,
            participating_clients=20,
            aggregation_time=5.5,
            privacy_budget_spent=0.5,
        )
        formatted = format_federated_stats(stats)
        assert "Round: 10" in formatted
        assert "Clients: 20" in formatted
        assert "5.50s" in formatted
        assert "0.5000" in formatted

    def test_contains_all_fields(self) -> None:
        """Contains all fields."""
        stats = create_federated_stats()
        formatted = format_federated_stats(stats)
        assert "Federated Stats" in formatted
        assert "Round:" in formatted
        assert "Clients:" in formatted
        assert "Aggregation Time:" in formatted
        assert "Privacy Budget Spent:" in formatted


class TestGetRecommendedFederatedConfig:
    """Tests for get_recommended_federated_config function."""

    def test_small_deployment(self) -> None:
        """Small deployment config."""
        fed_cfg, _priv_cfg = get_recommended_federated_config(10)
        assert fed_cfg.num_clients == 10
        assert fed_cfg.client_fraction == 1.0
        assert fed_cfg.aggregation_method == AggregationMethod.FEDAVG

    def test_medium_deployment(self) -> None:
        """Medium deployment config."""
        fed_cfg, _priv_cfg = get_recommended_federated_config(100)
        assert fed_cfg.num_clients == 100
        assert fed_cfg.client_fraction == 0.1

    def test_large_deployment(self) -> None:
        """Large deployment uses FedProx."""
        fed_cfg, _priv_cfg = get_recommended_federated_config(1000)
        assert fed_cfg.aggregation_method == AggregationMethod.FEDPROX

    def test_no_privacy_default(self) -> None:
        """Default is no privacy."""
        _fed_cfg, priv_cfg = get_recommended_federated_config(100)
        assert priv_cfg.mechanism == PrivacyMechanism.NONE

    def test_privacy_required(self) -> None:
        """Privacy required uses local DP."""
        _fed_cfg, priv_cfg = get_recommended_federated_config(
            100, privacy_required=True
        )
        assert priv_cfg.mechanism == PrivacyMechanism.LOCAL_DP

    def test_zero_clients_raises(self) -> None:
        """Zero clients raises ValueError."""
        with pytest.raises(ValueError, match="num_clients must be positive"):
            get_recommended_federated_config(0)


class TestValidConstants:
    """Tests for validation constants."""

    def test_valid_aggregation_methods_frozenset(self) -> None:
        """VALID_AGGREGATION_METHODS is a frozenset."""
        assert isinstance(VALID_AGGREGATION_METHODS, frozenset)
        assert len(VALID_AGGREGATION_METHODS) == 4

    def test_valid_client_selection_strategies_frozenset(self) -> None:
        """VALID_CLIENT_SELECTION_STRATEGIES is a frozenset."""
        assert isinstance(VALID_CLIENT_SELECTION_STRATEGIES, frozenset)
        assert len(VALID_CLIENT_SELECTION_STRATEGIES) == 4

    def test_valid_privacy_mechanisms_frozenset(self) -> None:
        """VALID_PRIVACY_MECHANISMS is a frozenset."""
        assert isinstance(VALID_PRIVACY_MECHANISMS, frozenset)
        assert len(VALID_PRIVACY_MECHANISMS) == 4
