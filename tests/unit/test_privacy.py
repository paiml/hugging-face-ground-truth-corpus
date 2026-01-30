"""Tests for differential privacy and data anonymization functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.safety.privacy import (
    VALID_ANONYMIZATION_TYPES,
    VALID_PRIVACY_MECHANISMS,
    VALID_SENSITIVITY_TYPES,
    AnonymizationConfig,
    AnonymizationType,
    DPConfig,
    PrivacyConfig,
    PrivacyMechanism,
    PrivacyStats,
    SensitivityType,
    add_differential_privacy_noise,
    calculate_noise_scale,
    check_k_anonymity,
    compute_privacy_budget,
    create_anonymization_config,
    create_dp_config,
    create_privacy_config,
    estimate_utility_loss,
    format_privacy_stats,
    get_anonymization_type,
    get_privacy_mechanism,
    get_recommended_privacy_config,
    get_sensitivity_type,
    list_anonymization_types,
    list_privacy_mechanisms,
    list_sensitivity_types,
    validate_anonymization_config,
    validate_dp_config,
    validate_privacy_config,
)


class TestPrivacyMechanism:
    """Tests for PrivacyMechanism enum."""

    def test_laplace_value(self) -> None:
        """Test LAPLACE value."""
        assert PrivacyMechanism.LAPLACE.value == "laplace"

    def test_gaussian_value(self) -> None:
        """Test GAUSSIAN value."""
        assert PrivacyMechanism.GAUSSIAN.value == "gaussian"

    def test_exponential_value(self) -> None:
        """Test EXPONENTIAL value."""
        assert PrivacyMechanism.EXPONENTIAL.value == "exponential"

    def test_randomized_response_value(self) -> None:
        """Test RANDOMIZED_RESPONSE value."""
        assert PrivacyMechanism.RANDOMIZED_RESPONSE.value == "randomized_response"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [m.value for m in PrivacyMechanism]
        assert len(values) == len(set(values))


class TestAnonymizationType:
    """Tests for AnonymizationType enum."""

    def test_k_anonymity_value(self) -> None:
        """Test K_ANONYMITY value."""
        assert AnonymizationType.K_ANONYMITY.value == "k_anonymity"

    def test_l_diversity_value(self) -> None:
        """Test L_DIVERSITY value."""
        assert AnonymizationType.L_DIVERSITY.value == "l_diversity"

    def test_t_closeness_value(self) -> None:
        """Test T_CLOSENESS value."""
        assert AnonymizationType.T_CLOSENESS.value == "t_closeness"

    def test_differential_value(self) -> None:
        """Test DIFFERENTIAL value."""
        assert AnonymizationType.DIFFERENTIAL.value == "differential"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [a.value for a in AnonymizationType]
        assert len(values) == len(set(values))


class TestSensitivityType:
    """Tests for SensitivityType enum."""

    def test_local_value(self) -> None:
        """Test LOCAL value."""
        assert SensitivityType.LOCAL.value == "local"

    def test_global_value(self) -> None:
        """Test GLOBAL value."""
        assert SensitivityType.GLOBAL.value == "global"

    def test_smooth_value(self) -> None:
        """Test SMOOTH value."""
        assert SensitivityType.SMOOTH.value == "smooth"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [s.value for s in SensitivityType]
        assert len(values) == len(set(values))


class TestValidFrozensets:
    """Tests for VALID_* frozenset constants."""

    def test_valid_privacy_mechanisms_contains_all_enums(self) -> None:
        """Test VALID_PRIVACY_MECHANISMS contains all enum values."""
        for m in PrivacyMechanism:
            assert m.value in VALID_PRIVACY_MECHANISMS

    def test_valid_anonymization_types_contains_all_enums(self) -> None:
        """Test VALID_ANONYMIZATION_TYPES contains all enum values."""
        for a in AnonymizationType:
            assert a.value in VALID_ANONYMIZATION_TYPES

    def test_valid_sensitivity_types_contains_all_enums(self) -> None:
        """Test VALID_SENSITIVITY_TYPES contains all enum values."""
        for s in SensitivityType:
            assert s.value in VALID_SENSITIVITY_TYPES

    def test_frozensets_are_immutable(self) -> None:
        """Test that frozensets cannot be modified."""
        with pytest.raises(AttributeError):
            VALID_PRIVACY_MECHANISMS.add("new")  # type: ignore[attr-defined]


class TestDPConfig:
    """Tests for DPConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating DPConfig instance."""
        config = DPConfig(
            epsilon=0.5,
            delta=1e-6,
            mechanism=PrivacyMechanism.GAUSSIAN,
            clip_norm=2.0,
            noise_multiplier=1.5,
        )
        assert config.epsilon == 0.5
        assert config.delta == 1e-6
        assert config.mechanism == PrivacyMechanism.GAUSSIAN
        assert config.clip_norm == 2.0
        assert config.noise_multiplier == 1.5

    def test_default_values(self) -> None:
        """Test default values for DPConfig."""
        config = DPConfig()
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.mechanism == PrivacyMechanism.LAPLACE
        assert config.clip_norm == 1.0
        assert config.noise_multiplier == 1.0

    def test_frozen(self) -> None:
        """Test that DPConfig is immutable."""
        config = DPConfig()
        with pytest.raises(AttributeError):
            config.epsilon = 0.5  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that DPConfig uses slots."""
        config = DPConfig()
        assert not hasattr(config, "__dict__")


class TestAnonymizationConfig:
    """Tests for AnonymizationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating AnonymizationConfig instance."""
        config = AnonymizationConfig(
            anon_type=AnonymizationType.L_DIVERSITY,
            k_value=10,
            quasi_identifiers=("age", "zipcode"),
            sensitive_attributes=("salary", "disease"),
        )
        assert config.anon_type == AnonymizationType.L_DIVERSITY
        assert config.k_value == 10
        assert "age" in config.quasi_identifiers
        assert "salary" in config.sensitive_attributes

    def test_default_values(self) -> None:
        """Test default values for AnonymizationConfig."""
        config = AnonymizationConfig()
        assert config.anon_type == AnonymizationType.K_ANONYMITY
        assert config.k_value == 5
        assert config.quasi_identifiers == ()
        assert config.sensitive_attributes == ()

    def test_frozen(self) -> None:
        """Test that AnonymizationConfig is immutable."""
        config = AnonymizationConfig()
        with pytest.raises(AttributeError):
            config.k_value = 10  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that AnonymizationConfig uses slots."""
        config = AnonymizationConfig()
        assert not hasattr(config, "__dict__")


class TestPrivacyConfig:
    """Tests for PrivacyConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PrivacyConfig instance."""
        dp = DPConfig(epsilon=0.5)
        anon = AnonymizationConfig(k_value=10)
        config = PrivacyConfig(
            dp_config=dp,
            anonymization_config=anon,
            audit_logging=True,
        )
        assert config.dp_config.epsilon == 0.5
        assert config.anonymization_config.k_value == 10
        assert config.audit_logging is True

    def test_default_values(self) -> None:
        """Test default values for PrivacyConfig."""
        config = PrivacyConfig()
        assert config.dp_config.epsilon == 1.0
        assert config.anonymization_config.k_value == 5
        assert config.audit_logging is False

    def test_frozen(self) -> None:
        """Test that PrivacyConfig is immutable."""
        config = PrivacyConfig()
        with pytest.raises(AttributeError):
            config.audit_logging = True  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that PrivacyConfig uses slots."""
        config = PrivacyConfig()
        assert not hasattr(config, "__dict__")


class TestPrivacyStats:
    """Tests for PrivacyStats dataclass."""

    def test_creation(self) -> None:
        """Test creating PrivacyStats instance."""
        stats = PrivacyStats(
            privacy_budget_spent=0.5,
            records_anonymized=1000,
            noise_added=42.5,
            utility_loss=0.15,
        )
        assert stats.privacy_budget_spent == 0.5
        assert stats.records_anonymized == 1000
        assert stats.noise_added == 42.5
        assert stats.utility_loss == 0.15

    def test_frozen(self) -> None:
        """Test that PrivacyStats is immutable."""
        stats = PrivacyStats(
            privacy_budget_spent=0.5,
            records_anonymized=1000,
            noise_added=42.5,
            utility_loss=0.15,
        )
        with pytest.raises(AttributeError):
            stats.privacy_budget_spent = 1.0  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that PrivacyStats uses slots."""
        stats = PrivacyStats(
            privacy_budget_spent=0.5,
            records_anonymized=1000,
            noise_added=42.5,
            utility_loss=0.15,
        )
        assert not hasattr(stats, "__dict__")


class TestValidateDPConfig:
    """Tests for validate_dp_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = DPConfig(epsilon=1.0, delta=1e-5)
        validate_dp_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_dp_config(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("epsilon", [0.0, -0.1, -1.0])
    def test_invalid_epsilon_raises_error(self, epsilon: float) -> None:
        """Test that non-positive epsilon raises ValueError."""
        config = DPConfig(epsilon=epsilon)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            validate_dp_config(config)

    @pytest.mark.parametrize("delta", [-0.1, 1.0, 1.5, 2.0])
    def test_invalid_delta_raises_error(self, delta: float) -> None:
        """Test that invalid delta raises ValueError."""
        config = DPConfig(delta=delta)
        with pytest.raises(ValueError, match=r"delta must be in \[0, 1\)"):
            validate_dp_config(config)

    @pytest.mark.parametrize("clip_norm", [0.0, -0.1, -1.0])
    def test_invalid_clip_norm_raises_error(self, clip_norm: float) -> None:
        """Test that non-positive clip_norm raises ValueError."""
        config = DPConfig(clip_norm=clip_norm)
        with pytest.raises(ValueError, match="clip_norm must be positive"):
            validate_dp_config(config)

    @pytest.mark.parametrize("noise_multiplier", [0.0, -0.1, -1.0])
    def test_invalid_noise_multiplier_raises_error(
        self, noise_multiplier: float
    ) -> None:
        """Test that non-positive noise_multiplier raises ValueError."""
        config = DPConfig(noise_multiplier=noise_multiplier)
        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            validate_dp_config(config)

    def test_delta_zero_is_valid(self) -> None:
        """Test that delta=0 is valid (pure DP)."""
        config = DPConfig(delta=0.0)
        validate_dp_config(config)  # Should not raise


class TestValidateAnonymizationConfig:
    """Tests for validate_anonymization_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = AnonymizationConfig(k_value=5)
        validate_anonymization_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_anonymization_config(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("k_value", [0, 1, -1, -5])
    def test_invalid_k_value_raises_error(self, k_value: int) -> None:
        """Test that k_value < 2 raises ValueError."""
        config = AnonymizationConfig(k_value=k_value)
        with pytest.raises(ValueError, match="k_value must be at least 2"):
            validate_anonymization_config(config)

    def test_k_value_2_is_valid(self) -> None:
        """Test that k_value=2 is valid."""
        config = AnonymizationConfig(k_value=2)
        validate_anonymization_config(config)  # Should not raise


class TestValidatePrivacyConfig:
    """Tests for validate_privacy_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = PrivacyConfig()
        validate_privacy_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_privacy_config(None)  # type: ignore[arg-type]


class TestCreateDPConfig:
    """Tests for create_dp_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_dp_config()
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.mechanism == PrivacyMechanism.LAPLACE

    def test_create_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_dp_config(
            epsilon=0.5,
            delta=1e-6,
            mechanism=PrivacyMechanism.GAUSSIAN,
            clip_norm=2.0,
            noise_multiplier=1.5,
        )
        assert config.epsilon == 0.5
        assert config.delta == 1e-6
        assert config.mechanism == PrivacyMechanism.GAUSSIAN
        assert config.clip_norm == 2.0
        assert config.noise_multiplier == 1.5

    def test_create_with_string_mechanism(self) -> None:
        """Test creating config with string mechanism."""
        config = create_dp_config(mechanism="gaussian")
        assert config.mechanism == PrivacyMechanism.GAUSSIAN

    def test_invalid_epsilon_raises_error(self) -> None:
        """Test that invalid epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            create_dp_config(epsilon=0)

    def test_invalid_mechanism_raises_error(self) -> None:
        """Test that invalid mechanism raises ValueError."""
        with pytest.raises(ValueError, match="invalid privacy mechanism"):
            create_dp_config(mechanism="invalid")


class TestCreateAnonymizationConfig:
    """Tests for create_anonymization_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_anonymization_config()
        assert config.anon_type == AnonymizationType.K_ANONYMITY
        assert config.k_value == 5
        assert config.quasi_identifiers == ()
        assert config.sensitive_attributes == ()

    def test_create_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_anonymization_config(
            anon_type=AnonymizationType.L_DIVERSITY,
            k_value=10,
            quasi_identifiers=("age", "zipcode"),
            sensitive_attributes=("salary",),
        )
        assert config.anon_type == AnonymizationType.L_DIVERSITY
        assert config.k_value == 10
        assert "age" in config.quasi_identifiers
        assert "salary" in config.sensitive_attributes

    def test_create_with_string_type(self) -> None:
        """Test creating config with string type."""
        config = create_anonymization_config(anon_type="l_diversity")
        assert config.anon_type == AnonymizationType.L_DIVERSITY

    def test_create_with_list_identifiers(self) -> None:
        """Test creating config with list identifiers."""
        config = create_anonymization_config(quasi_identifiers=["age", "zipcode"])
        assert config.quasi_identifiers == ("age", "zipcode")

    def test_create_with_list_attributes(self) -> None:
        """Test creating config with list attributes."""
        config = create_anonymization_config(sensitive_attributes=["salary", "disease"])
        assert config.sensitive_attributes == ("salary", "disease")

    def test_invalid_k_value_raises_error(self) -> None:
        """Test that invalid k_value raises ValueError."""
        with pytest.raises(ValueError, match="k_value must be at least 2"):
            create_anonymization_config(k_value=1)

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid anonymization type"):
            create_anonymization_config(anon_type="invalid")


class TestCreatePrivacyConfig:
    """Tests for create_privacy_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_privacy_config()
        assert config.dp_config.epsilon == 1.0
        assert config.anonymization_config.k_value == 5
        assert config.audit_logging is False

    def test_create_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        dp = create_dp_config(epsilon=0.5)
        anon = create_anonymization_config(k_value=10)
        config = create_privacy_config(
            dp_config=dp,
            anonymization_config=anon,
            audit_logging=True,
        )
        assert config.dp_config.epsilon == 0.5
        assert config.anonymization_config.k_value == 10
        assert config.audit_logging is True


class TestListPrivacyMechanisms:
    """Tests for list_privacy_mechanisms function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        mechanisms = list_privacy_mechanisms()
        assert isinstance(mechanisms, list)

    def test_contains_expected_mechanisms(self) -> None:
        """Test that list contains expected mechanisms."""
        mechanisms = list_privacy_mechanisms()
        assert "laplace" in mechanisms
        assert "gaussian" in mechanisms
        assert "exponential" in mechanisms
        assert "randomized_response" in mechanisms

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        mechanisms = list_privacy_mechanisms()
        assert mechanisms == sorted(mechanisms)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        mechanisms = list_privacy_mechanisms()
        assert all(isinstance(m, str) for m in mechanisms)


class TestListAnonymizationTypes:
    """Tests for list_anonymization_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_anonymization_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_anonymization_types()
        assert "k_anonymity" in types
        assert "l_diversity" in types
        assert "t_closeness" in types
        assert "differential" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_anonymization_types()
        assert types == sorted(types)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        types = list_anonymization_types()
        assert all(isinstance(t, str) for t in types)


class TestListSensitivityTypes:
    """Tests for list_sensitivity_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_sensitivity_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_sensitivity_types()
        assert "local" in types
        assert "global" in types
        assert "smooth" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_sensitivity_types()
        assert types == sorted(types)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        types = list_sensitivity_types()
        assert all(isinstance(t, str) for t in types)


class TestGetPrivacyMechanism:
    """Tests for get_privacy_mechanism function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("laplace", PrivacyMechanism.LAPLACE),
            ("gaussian", PrivacyMechanism.GAUSSIAN),
            ("exponential", PrivacyMechanism.EXPONENTIAL),
            ("randomized_response", PrivacyMechanism.RANDOMIZED_RESPONSE),
        ],
    )
    def test_valid_names(self, name: str, expected: PrivacyMechanism) -> None:
        """Test getting mechanism by valid name."""
        assert get_privacy_mechanism(name) == expected

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="invalid privacy mechanism"):
            get_privacy_mechanism("invalid")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="invalid privacy mechanism"):
            get_privacy_mechanism("")


class TestGetAnonymizationType:
    """Tests for get_anonymization_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("k_anonymity", AnonymizationType.K_ANONYMITY),
            ("l_diversity", AnonymizationType.L_DIVERSITY),
            ("t_closeness", AnonymizationType.T_CLOSENESS),
            ("differential", AnonymizationType.DIFFERENTIAL),
        ],
    )
    def test_valid_names(self, name: str, expected: AnonymizationType) -> None:
        """Test getting type by valid name."""
        assert get_anonymization_type(name) == expected

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="invalid anonymization type"):
            get_anonymization_type("invalid")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="invalid anonymization type"):
            get_anonymization_type("")


class TestGetSensitivityType:
    """Tests for get_sensitivity_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("local", SensitivityType.LOCAL),
            ("global", SensitivityType.GLOBAL),
            ("smooth", SensitivityType.SMOOTH),
        ],
    )
    def test_valid_names(self, name: str, expected: SensitivityType) -> None:
        """Test getting type by valid name."""
        assert get_sensitivity_type(name) == expected

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="invalid sensitivity type"):
            get_sensitivity_type("invalid")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="invalid sensitivity type"):
            get_sensitivity_type("")


class TestCalculateNoiseScale:
    """Tests for calculate_noise_scale function."""

    def test_laplace_basic(self) -> None:
        """Test Laplace mechanism basic calculation."""
        scale = calculate_noise_scale(epsilon=1.0, sensitivity=1.0)
        assert scale == 1.0

    def test_laplace_higher_sensitivity(self) -> None:
        """Test Laplace with higher sensitivity."""
        scale = calculate_noise_scale(epsilon=0.5, sensitivity=2.0)
        assert scale == 4.0

    def test_laplace_lower_epsilon(self) -> None:
        """Test Laplace with lower epsilon (more privacy)."""
        scale = calculate_noise_scale(epsilon=0.1, sensitivity=1.0)
        assert scale == 10.0

    def test_gaussian_mechanism(self) -> None:
        """Test Gaussian mechanism calculation."""
        scale = calculate_noise_scale(
            epsilon=1.0,
            sensitivity=1.0,
            delta=1e-5,
            mechanism=PrivacyMechanism.GAUSSIAN,
        )
        assert scale > 0
        # Gaussian scale should be higher than Laplace for same epsilon
        laplace_scale = calculate_noise_scale(epsilon=1.0, sensitivity=1.0)
        assert scale > laplace_scale

    @pytest.mark.parametrize("epsilon", [0.0, -0.1, -1.0])
    def test_invalid_epsilon_raises_error(self, epsilon: float) -> None:
        """Test that non-positive epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            calculate_noise_scale(epsilon=epsilon, sensitivity=1.0)

    @pytest.mark.parametrize("sensitivity", [0.0, -0.1, -1.0])
    def test_invalid_sensitivity_raises_error(self, sensitivity: float) -> None:
        """Test that non-positive sensitivity raises ValueError."""
        with pytest.raises(ValueError, match="sensitivity must be positive"):
            calculate_noise_scale(epsilon=1.0, sensitivity=sensitivity)

    @pytest.mark.parametrize("delta", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_delta_for_gaussian_raises_error(self, delta: float) -> None:
        """Test that invalid delta for Gaussian raises ValueError."""
        with pytest.raises(ValueError, match="delta must be in"):
            calculate_noise_scale(
                epsilon=1.0,
                sensitivity=1.0,
                delta=delta,
                mechanism=PrivacyMechanism.GAUSSIAN,
            )


class TestComputePrivacyBudget:
    """Tests for compute_privacy_budget function."""

    def test_simple_composition(self) -> None:
        """Test simple composition (sum)."""
        budget = compute_privacy_budget(10, 0.1, composition="simple")
        assert budget == 1.0

    def test_advanced_composition(self) -> None:
        """Test advanced composition (sqrt)."""
        budget = compute_privacy_budget(10, 0.1, composition="advanced")
        assert budget == pytest.approx(0.316227766, rel=1e-5)  # sqrt(10) * 0.1

    def test_advanced_is_tighter(self) -> None:
        """Test that advanced composition is tighter than simple."""
        simple = compute_privacy_budget(100, 0.1, composition="simple")
        advanced = compute_privacy_budget(100, 0.1, composition="advanced")
        assert advanced < simple

    @pytest.mark.parametrize("num_queries", [0, -1, -10])
    def test_invalid_num_queries_raises_error(self, num_queries: int) -> None:
        """Test that non-positive num_queries raises ValueError."""
        with pytest.raises(ValueError, match="num_queries must be positive"):
            compute_privacy_budget(num_queries, 0.1)

    @pytest.mark.parametrize("epsilon", [0.0, -0.1, -1.0])
    def test_invalid_epsilon_raises_error(self, epsilon: float) -> None:
        """Test that non-positive epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon_per_query must be positive"):
            compute_privacy_budget(10, epsilon)

    def test_invalid_composition_raises_error(self) -> None:
        """Test that invalid composition raises ValueError."""
        with pytest.raises(ValueError, match="composition must be"):
            compute_privacy_budget(10, 0.1, composition="invalid")


class TestAddDifferentialPrivacyNoise:
    """Tests for add_differential_privacy_noise function."""

    def test_adds_noise(self) -> None:
        """Test that noise is added to value."""
        noisy = add_differential_privacy_noise(100.0, epsilon=1.0, sensitivity=1.0)
        assert noisy != 100.0

    def test_noise_proportional_to_sensitivity(self) -> None:
        """Test that more sensitivity = more noise."""
        noisy_low = add_differential_privacy_noise(100.0, epsilon=1.0, sensitivity=1.0)
        noisy_high = add_differential_privacy_noise(100.0, epsilon=1.0, sensitivity=2.0)
        # Higher sensitivity should result in larger deviation from original
        assert abs(noisy_high - 100.0) > abs(noisy_low - 100.0)

    def test_noise_inversely_proportional_to_epsilon(self) -> None:
        """Test that lower epsilon = more noise."""
        noisy_high_eps = add_differential_privacy_noise(
            100.0, epsilon=10.0, sensitivity=1.0
        )
        noisy_low_eps = add_differential_privacy_noise(
            100.0, epsilon=0.1, sensitivity=1.0
        )
        # Lower epsilon should result in larger deviation from original
        assert abs(noisy_low_eps - 100.0) > abs(noisy_high_eps - 100.0)

    @pytest.mark.parametrize("epsilon", [0.0, -0.1, -1.0])
    def test_invalid_epsilon_raises_error(self, epsilon: float) -> None:
        """Test that non-positive epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            add_differential_privacy_noise(100.0, epsilon=epsilon, sensitivity=1.0)


class TestCheckKAnonymity:
    """Tests for check_k_anonymity function."""

    def test_satisfied(self) -> None:
        """Test k-anonymity satisfied."""
        assert check_k_anonymity([5, 6, 7, 8], k=5) is True

    def test_not_satisfied(self) -> None:
        """Test k-anonymity not satisfied."""
        assert check_k_anonymity([3, 5, 6], k=5) is False

    def test_exactly_k(self) -> None:
        """Test with exactly k in all groups."""
        assert check_k_anonymity([5, 5, 5], k=5) is True

    def test_single_group_satisfied(self) -> None:
        """Test single group that satisfies."""
        assert check_k_anonymity([10], k=5) is True

    def test_single_group_not_satisfied(self) -> None:
        """Test single group that doesn't satisfy."""
        assert check_k_anonymity([3], k=5) is False

    def test_empty_group_sizes_raises_error(self) -> None:
        """Test that empty group_sizes raises ValueError."""
        with pytest.raises(ValueError, match="group_sizes cannot be empty"):
            check_k_anonymity([], k=5)

    @pytest.mark.parametrize("k", [0, 1, -1, -5])
    def test_invalid_k_raises_error(self, k: int) -> None:
        """Test that k < 2 raises ValueError."""
        with pytest.raises(ValueError, match="k must be at least 2"):
            check_k_anonymity([5, 6], k=k)

    def test_k_equals_2_is_valid(self) -> None:
        """Test that k=2 is valid."""
        assert check_k_anonymity([2, 3, 4], k=2) is True


class TestEstimateUtilityLoss:
    """Tests for estimate_utility_loss function."""

    def test_returns_between_0_and_1(self) -> None:
        """Test that utility loss is in [0, 1]."""
        loss = estimate_utility_loss(epsilon=1.0, sensitivity=1.0)
        assert 0 <= loss <= 1

    def test_higher_epsilon_lower_loss(self) -> None:
        """Test that higher epsilon = lower utility loss."""
        loss_high_eps = estimate_utility_loss(epsilon=10.0, sensitivity=1.0)
        loss_low_eps = estimate_utility_loss(epsilon=0.1, sensitivity=1.0)
        assert loss_high_eps < loss_low_eps

    def test_higher_sensitivity_higher_loss(self) -> None:
        """Test that higher sensitivity = higher utility loss."""
        loss_low_sens = estimate_utility_loss(epsilon=1.0, sensitivity=0.5)
        loss_high_sens = estimate_utility_loss(epsilon=1.0, sensitivity=2.0)
        assert loss_high_sens > loss_low_sens

    @pytest.mark.parametrize("epsilon", [0.0, -0.1, -1.0])
    def test_invalid_epsilon_raises_error(self, epsilon: float) -> None:
        """Test that non-positive epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            estimate_utility_loss(epsilon=epsilon, sensitivity=1.0)


class TestFormatPrivacyStats:
    """Tests for format_privacy_stats function."""

    def test_format_basic(self) -> None:
        """Test basic formatting."""
        stats = PrivacyStats(
            privacy_budget_spent=0.5,
            records_anonymized=1000,
            noise_added=42.5,
            utility_loss=0.15,
        )
        formatted = format_privacy_stats(stats)
        assert "Privacy Budget Spent: 0.50" in formatted
        assert "Records Anonymized: 1000" in formatted
        assert "Noise Added: 42.50" in formatted
        assert "Utility Loss: 15.0%" in formatted

    def test_format_zero_values(self) -> None:
        """Test formatting with zero values."""
        stats = PrivacyStats(
            privacy_budget_spent=0.0,
            records_anonymized=0,
            noise_added=0.0,
            utility_loss=0.0,
        )
        formatted = format_privacy_stats(stats)
        assert "Privacy Budget Spent: 0.00" in formatted
        assert "Records Anonymized: 0" in formatted
        assert "Noise Added: 0.00" in formatted
        assert "Utility Loss: 0.0%" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_privacy_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedPrivacyConfig:
    """Tests for get_recommended_privacy_config function."""

    def test_training_high_sensitivity(self) -> None:
        """Test training with high sensitivity."""
        config = get_recommended_privacy_config("training", "high")
        assert config.dp_config.epsilon < 1.0
        assert config.anonymization_config.k_value >= 5
        assert config.dp_config.mechanism == PrivacyMechanism.GAUSSIAN
        assert config.audit_logging is True

    def test_training_medium_sensitivity(self) -> None:
        """Test training with medium sensitivity."""
        config = get_recommended_privacy_config("training", "medium")
        assert config.dp_config.epsilon == 0.5
        assert config.anonymization_config.k_value == 5

    def test_inference_low_sensitivity(self) -> None:
        """Test inference with low sensitivity."""
        config = get_recommended_privacy_config("inference", "low")
        assert config.dp_config.epsilon >= 1.0
        assert config.dp_config.mechanism == PrivacyMechanism.LAPLACE
        assert config.audit_logging is False

    def test_analysis_medium_sensitivity(self) -> None:
        """Test analysis with medium sensitivity."""
        config = get_recommended_privacy_config("analysis", "medium")
        assert config.dp_config.epsilon > 1.0  # Analysis is more lenient

    def test_invalid_use_case_raises_error(self) -> None:
        """Test that invalid use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_privacy_config("invalid", "medium")

    def test_invalid_sensitivity_raises_error(self) -> None:
        """Test that invalid sensitivity raises ValueError."""
        with pytest.raises(ValueError, match="data_sensitivity must be one of"):
            get_recommended_privacy_config("training", "invalid")

    @pytest.mark.parametrize("use_case", ["training", "inference", "analysis"])
    @pytest.mark.parametrize("sensitivity", ["low", "medium", "high"])
    def test_all_combinations_valid(self, use_case: str, sensitivity: str) -> None:
        """Test all valid combinations work."""
        config = get_recommended_privacy_config(use_case, sensitivity)
        validate_privacy_config(config)  # Should not raise


class TestHypothesis:
    """Property-based tests using Hypothesis."""

    @given(st.floats(min_value=0.01, max_value=100.0))
    @settings(max_examples=50)
    def test_noise_scale_positive(self, epsilon: float) -> None:
        """Test that noise scale is always positive."""
        scale = calculate_noise_scale(epsilon=epsilon, sensitivity=1.0)
        assert scale > 0

    @given(
        st.integers(min_value=1, max_value=1000),
        st.floats(min_value=0.01, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_budget_composition_valid(
        self, num_queries: int, epsilon_per_query: float
    ) -> None:
        """Test that budget computation is valid."""
        simple = compute_privacy_budget(num_queries, epsilon_per_query, "simple")
        advanced = compute_privacy_budget(num_queries, epsilon_per_query, "advanced")
        assert simple > 0
        assert advanced > 0
        assert advanced <= simple

    @given(st.floats(min_value=0.01, max_value=100.0))
    @settings(max_examples=50)
    def test_utility_loss_in_range(self, epsilon: float) -> None:
        """Test that utility loss is always in [0, 1]."""
        loss = estimate_utility_loss(epsilon=epsilon, sensitivity=1.0)
        assert 0 <= loss <= 1

    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_k_anonymity_consistent(self, group_sizes: list[int]) -> None:
        """Test k-anonymity check is consistent."""
        min_size = min(group_sizes)
        # If k <= min_size, should pass; if k > min_size, should fail
        if min_size >= 2:
            assert check_k_anonymity(group_sizes, k=2) is True
        # k > all sizes should fail
        max_k = max(group_sizes) + 1
        if max_k >= 2:
            assert check_k_anonymity(group_sizes, k=max_k) is False

    @given(st.sampled_from(list(PrivacyMechanism)))
    def test_all_mechanisms_have_string_value(
        self, mechanism: PrivacyMechanism
    ) -> None:
        """Test that all mechanisms have string values."""
        result = get_privacy_mechanism(mechanism.value)
        assert result == mechanism

    @given(st.sampled_from(list(AnonymizationType)))
    def test_all_anon_types_have_string_value(
        self, anon_type: AnonymizationType
    ) -> None:
        """Test that all anonymization types have string values."""
        result = get_anonymization_type(anon_type.value)
        assert result == anon_type

    @given(st.sampled_from(list(SensitivityType)))
    def test_all_sens_types_have_string_value(self, sens_type: SensitivityType) -> None:
        """Test that all sensitivity types have string values."""
        result = get_sensitivity_type(sens_type.value)
        assert result == sens_type
