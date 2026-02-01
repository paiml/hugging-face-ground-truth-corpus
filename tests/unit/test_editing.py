"""Tests for model editing functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.editing import (
    EditingConfig,
    EditingMethod,
    EditingStats,
    EditRequest,
    EditScope,
    LocalizationResult,
    LocalizationType,
    MEMITConfig,
    ROMEConfig,
    calculate_edit_success,
    create_edit_request,
    create_editing_config,
    create_memit_config,
    create_rome_config,
    format_editing_stats,
    get_edit_scope,
    get_editing_method,
    get_localization_type,
    get_recommended_editing_config,
    list_edit_scopes,
    list_editing_methods,
    list_localization_types,
    localize_knowledge,
    measure_generalization,
    measure_specificity,
    validate_edit_request,
    validate_edit_scope,
    validate_editing_config,
    validate_editing_method,
    validate_localization_type,
    validate_memit_config,
    validate_rome_config,
)


class TestEditingMethod:
    """Tests for EditingMethod enum."""

    def test_rome_value(self) -> None:
        """Test ROME value."""
        assert EditingMethod.ROME.value == "rome"

    def test_memit_value(self) -> None:
        """Test MEMIT value."""
        assert EditingMethod.MEMIT.value == "memit"

    def test_mend_value(self) -> None:
        """Test MEND value."""
        assert EditingMethod.MEND.value == "mend"

    def test_serac_value(self) -> None:
        """Test SERAC value."""
        assert EditingMethod.SERAC.value == "serac"

    def test_ft_value(self) -> None:
        """Test FT value."""
        assert EditingMethod.FT.value == "ft"


class TestLocalizationType:
    """Tests for LocalizationType enum."""

    def test_causal_tracing_value(self) -> None:
        """Test CAUSAL_TRACING value."""
        assert LocalizationType.CAUSAL_TRACING.value == "causal_tracing"

    def test_activation_patching_value(self) -> None:
        """Test ACTIVATION_PATCHING value."""
        assert LocalizationType.ACTIVATION_PATCHING.value == "activation_patching"

    def test_gradient_based_value(self) -> None:
        """Test GRADIENT_BASED value."""
        assert LocalizationType.GRADIENT_BASED.value == "gradient_based"


class TestEditScope:
    """Tests for EditScope enum."""

    def test_single_value(self) -> None:
        """Test SINGLE value."""
        assert EditScope.SINGLE.value == "single"

    def test_batch_value(self) -> None:
        """Test BATCH value."""
        assert EditScope.BATCH.value == "batch"

    def test_sequential_value(self) -> None:
        """Test SEQUENTIAL value."""
        assert EditScope.SEQUENTIAL.value == "sequential"


class TestROMEConfig:
    """Tests for ROMEConfig dataclass."""

    def test_required_field(self) -> None:
        """Test that layers is required."""
        config = ROMEConfig(layers=[5, 6, 7])
        assert config.layers == [5, 6, 7]

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ROMEConfig(layers=[5])
        assert config.v_lr == pytest.approx(0.5)
        assert config.clamp_norm == pytest.approx(4.0)
        assert config.kl_factor == pytest.approx(0.0625)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ROMEConfig(
            layers=[10, 11],
            v_lr=0.3,
            clamp_norm=2.0,
            kl_factor=0.1,
        )
        assert config.layers == [10, 11]
        assert config.v_lr == pytest.approx(0.3)
        assert config.clamp_norm == pytest.approx(2.0)
        assert config.kl_factor == pytest.approx(0.1)

    def test_frozen(self) -> None:
        """Test that ROMEConfig is immutable."""
        config = ROMEConfig(layers=[5])
        with pytest.raises(AttributeError):
            config.v_lr = 0.1  # type: ignore[misc]


class TestMEMITConfig:
    """Tests for MEMITConfig dataclass."""

    def test_required_field(self) -> None:
        """Test that layers is required."""
        config = MEMITConfig(layers=[4, 5, 6])
        assert config.layers == [4, 5, 6]

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MEMITConfig(layers=[5])
        assert config.lambda_weight == pytest.approx(5000.0)
        assert config.edit_weight == pytest.approx(1.0)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MEMITConfig(
            layers=[8, 9, 10],
            lambda_weight=10000.0,
            edit_weight=0.5,
        )
        assert config.layers == [8, 9, 10]
        assert config.lambda_weight == pytest.approx(10000.0)
        assert config.edit_weight == pytest.approx(0.5)

    def test_frozen(self) -> None:
        """Test that MEMITConfig is immutable."""
        config = MEMITConfig(layers=[5])
        with pytest.raises(AttributeError):
            config.lambda_weight = 1000.0  # type: ignore[misc]


class TestEditRequest:
    """Tests for EditRequest dataclass."""

    def test_creation(self) -> None:
        """Test creating EditRequest instance."""
        request = EditRequest(
            subject="The Eiffel Tower",
            target="Rome",
            prompt="{} is located in",
            ground_truth="Paris",
        )
        assert request.subject == "The Eiffel Tower"
        assert request.target == "Rome"
        assert request.prompt == "{} is located in"
        assert request.ground_truth == "Paris"

    def test_frozen(self) -> None:
        """Test that EditRequest is immutable."""
        request = EditRequest("a", "b", "c", "d")
        with pytest.raises(AttributeError):
            request.subject = "new"  # type: ignore[misc]


class TestEditingConfig:
    """Tests for EditingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating EditingConfig instance."""
        rome = ROMEConfig(layers=[5, 6])
        config = EditingConfig(
            method=EditingMethod.ROME,
            rome_config=rome,
            memit_config=None,
            scope=EditScope.SINGLE,
            verify_edit=True,
        )
        assert config.method == EditingMethod.ROME
        assert config.rome_config == rome
        assert config.scope == EditScope.SINGLE
        assert config.verify_edit is True

    def test_frozen(self) -> None:
        """Test that EditingConfig is immutable."""
        config = EditingConfig(
            method=EditingMethod.ROME,
            rome_config=None,
            memit_config=None,
            scope=EditScope.SINGLE,
            verify_edit=True,
        )
        with pytest.raises(AttributeError):
            config.verify_edit = False  # type: ignore[misc]


class TestEditingStats:
    """Tests for EditingStats dataclass."""

    def test_creation(self) -> None:
        """Test creating EditingStats instance."""
        stats = EditingStats(
            edit_success_rate=0.95,
            specificity=0.92,
            generalization=0.85,
            locality_score=0.88,
        )
        assert stats.edit_success_rate == pytest.approx(0.95)
        assert stats.specificity == pytest.approx(0.92)
        assert stats.generalization == pytest.approx(0.85)
        assert stats.locality_score == pytest.approx(0.88)

    def test_frozen(self) -> None:
        """Test that EditingStats is immutable."""
        stats = EditingStats(0.9, 0.9, 0.9, 0.9)
        with pytest.raises(AttributeError):
            stats.edit_success_rate = 0.5  # type: ignore[misc]


class TestLocalizationResult:
    """Tests for LocalizationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating LocalizationResult instance."""
        result = LocalizationResult(
            method=LocalizationType.CAUSAL_TRACING,
            layer_scores=[0.1, 0.5, 0.9],
            critical_layers=[1, 2],
            token_importance=[0.3, 0.7],
            metadata=None,
        )
        assert result.method == LocalizationType.CAUSAL_TRACING
        assert len(result.layer_scores) == 3
        assert result.critical_layers == [1, 2]

    def test_with_metadata(self) -> None:
        """Test creation with metadata."""
        result = LocalizationResult(
            method=LocalizationType.ACTIVATION_PATCHING,
            layer_scores=[0.5],
            critical_layers=[0],
            token_importance=[1.0],
            metadata={"key": "value"},
        )
        assert result.metadata == {"key": "value"}

    def test_frozen(self) -> None:
        """Test that LocalizationResult is immutable."""
        result = LocalizationResult(
            LocalizationType.CAUSAL_TRACING, [0.1], [0], [0.5], None
        )
        with pytest.raises(AttributeError):
            result.critical_layers = [1]  # type: ignore[misc]


class TestCreateROMEConfig:
    """Tests for create_rome_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating default config."""
        config = create_rome_config()
        assert config.layers == [5, 6, 7]
        assert config.v_lr == pytest.approx(0.5)

    def test_with_custom_layers(self) -> None:
        """Test with custom layers."""
        config = create_rome_config(layers=[10, 11, 12])
        assert config.layers == [10, 11, 12]

    def test_with_custom_v_lr(self) -> None:
        """Test with custom v_lr."""
        config = create_rome_config(v_lr=0.3)
        assert config.v_lr == pytest.approx(0.3)

    def test_with_custom_clamp_norm(self) -> None:
        """Test with custom clamp_norm."""
        config = create_rome_config(clamp_norm=2.0)
        assert config.clamp_norm == pytest.approx(2.0)

    def test_with_custom_kl_factor(self) -> None:
        """Test with custom kl_factor."""
        config = create_rome_config(kl_factor=0.1)
        assert config.kl_factor == pytest.approx(0.1)


class TestCreateMEMITConfig:
    """Tests for create_memit_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating default config."""
        config = create_memit_config()
        assert config.layers == [4, 5, 6, 7, 8]
        assert config.lambda_weight == pytest.approx(5000.0)

    def test_with_custom_layers(self) -> None:
        """Test with custom layers."""
        config = create_memit_config(layers=[8, 9, 10])
        assert config.layers == [8, 9, 10]

    def test_with_custom_lambda_weight(self) -> None:
        """Test with custom lambda_weight."""
        config = create_memit_config(lambda_weight=10000.0)
        assert config.lambda_weight == pytest.approx(10000.0)

    def test_with_custom_edit_weight(self) -> None:
        """Test with custom edit_weight."""
        config = create_memit_config(edit_weight=0.5)
        assert config.edit_weight == pytest.approx(0.5)


class TestCreateEditRequest:
    """Tests for create_edit_request function."""

    def test_creates_request(self) -> None:
        """Test creating an edit request."""
        req = create_edit_request(
            subject="The Eiffel Tower",
            target="Rome",
            prompt="{} is located in",
            ground_truth="Paris",
        )
        assert req.subject == "The Eiffel Tower"
        assert req.target == "Rome"
        assert req.prompt == "{} is located in"
        assert req.ground_truth == "Paris"


class TestCreateEditingConfig:
    """Tests for create_editing_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating default config."""
        config = create_editing_config()
        assert config.method == EditingMethod.ROME
        assert config.scope == EditScope.SINGLE
        assert config.verify_edit is True

    def test_with_rome_config(self) -> None:
        """Test with ROME config."""
        rome = create_rome_config(layers=[5, 6])
        config = create_editing_config(
            method=EditingMethod.ROME,
            rome_config=rome,
        )
        assert config.rome_config is not None
        assert config.rome_config.layers == [5, 6]

    def test_with_memit_config(self) -> None:
        """Test with MEMIT config."""
        memit = create_memit_config(layers=[4, 5])
        config = create_editing_config(
            method=EditingMethod.MEMIT,
            memit_config=memit,
        )
        assert config.memit_config is not None
        assert config.memit_config.layers == [4, 5]

    def test_with_batch_scope(self) -> None:
        """Test with batch scope."""
        config = create_editing_config(scope=EditScope.BATCH)
        assert config.scope == EditScope.BATCH

    def test_with_verify_edit_false(self) -> None:
        """Test with verify_edit=False."""
        config = create_editing_config(verify_edit=False)
        assert config.verify_edit is False


class TestValidateROMEConfig:
    """Tests for validate_rome_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_rome_config()
        validate_rome_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_rome_config(None)  # type: ignore[arg-type]

    def test_empty_layers_raises_error(self) -> None:
        """Test that empty layers raises ValueError."""
        config = ROMEConfig(layers=[])
        with pytest.raises(ValueError, match="layers cannot be empty"):
            validate_rome_config(config)

    def test_negative_layer_raises_error(self) -> None:
        """Test that negative layer raises ValueError."""
        config = ROMEConfig(layers=[-1, 5])
        with pytest.raises(ValueError, match="layers cannot contain negative"):
            validate_rome_config(config)

    def test_zero_v_lr_raises_error(self) -> None:
        """Test that zero v_lr raises ValueError."""
        config = ROMEConfig(layers=[5], v_lr=0)
        with pytest.raises(ValueError, match="v_lr must be positive"):
            validate_rome_config(config)

    def test_zero_clamp_norm_raises_error(self) -> None:
        """Test that zero clamp_norm raises ValueError."""
        config = ROMEConfig(layers=[5], clamp_norm=0)
        with pytest.raises(ValueError, match="clamp_norm must be positive"):
            validate_rome_config(config)

    def test_negative_kl_factor_raises_error(self) -> None:
        """Test that negative kl_factor raises ValueError."""
        config = ROMEConfig(layers=[5], kl_factor=-0.1)
        with pytest.raises(ValueError, match="kl_factor cannot be negative"):
            validate_rome_config(config)

    def test_zero_kl_factor_is_valid(self) -> None:
        """Test that zero kl_factor is valid."""
        config = ROMEConfig(layers=[5], kl_factor=0)
        validate_rome_config(config)  # Should not raise


class TestValidateMEMITConfig:
    """Tests for validate_memit_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_memit_config()
        validate_memit_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_memit_config(None)  # type: ignore[arg-type]

    def test_empty_layers_raises_error(self) -> None:
        """Test that empty layers raises ValueError."""
        config = MEMITConfig(layers=[])
        with pytest.raises(ValueError, match="layers cannot be empty"):
            validate_memit_config(config)

    def test_negative_layer_raises_error(self) -> None:
        """Test that negative layer raises ValueError."""
        config = MEMITConfig(layers=[-1, 5])
        with pytest.raises(ValueError, match="layers cannot contain negative"):
            validate_memit_config(config)

    def test_zero_lambda_weight_raises_error(self) -> None:
        """Test that zero lambda_weight raises ValueError."""
        config = MEMITConfig(layers=[5], lambda_weight=0)
        with pytest.raises(ValueError, match="lambda_weight must be positive"):
            validate_memit_config(config)

    def test_zero_edit_weight_raises_error(self) -> None:
        """Test that zero edit_weight raises ValueError."""
        config = MEMITConfig(layers=[5], edit_weight=0)
        with pytest.raises(ValueError, match="edit_weight must be positive"):
            validate_memit_config(config)


class TestValidateEditRequest:
    """Tests for validate_edit_request function."""

    def test_valid_request(self) -> None:
        """Test validation of valid request."""
        req = create_edit_request("Subject", "Target", "Prompt", "Ground")
        validate_edit_request(req)  # Should not raise

    def test_none_request_raises_error(self) -> None:
        """Test that None request raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_edit_request(None)  # type: ignore[arg-type]

    def test_empty_subject_raises_error(self) -> None:
        """Test that empty subject raises ValueError."""
        req = EditRequest(subject="", target="x", prompt="y", ground_truth="z")
        with pytest.raises(ValueError, match="subject cannot be empty"):
            validate_edit_request(req)

    def test_empty_target_raises_error(self) -> None:
        """Test that empty target raises ValueError."""
        req = EditRequest(subject="x", target="", prompt="y", ground_truth="z")
        with pytest.raises(ValueError, match="target cannot be empty"):
            validate_edit_request(req)

    def test_empty_prompt_raises_error(self) -> None:
        """Test that empty prompt raises ValueError."""
        req = EditRequest(subject="x", target="y", prompt="", ground_truth="z")
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            validate_edit_request(req)


class TestValidateEditingConfig:
    """Tests for validate_editing_config function."""

    def test_valid_rome_config(self) -> None:
        """Test validation of valid ROME config."""
        rome = create_rome_config()
        config = create_editing_config(method=EditingMethod.ROME, rome_config=rome)
        validate_editing_config(config)  # Should not raise

    def test_valid_memit_config(self) -> None:
        """Test validation of valid MEMIT config."""
        memit = create_memit_config()
        config = create_editing_config(method=EditingMethod.MEMIT, memit_config=memit)
        validate_editing_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_editing_config(None)  # type: ignore[arg-type]

    def test_rome_without_rome_config_raises_error(self) -> None:
        """Test that ROME without rome_config raises ValueError."""
        config = create_editing_config(method=EditingMethod.ROME, rome_config=None)
        with pytest.raises(ValueError, match="rome_config is required"):
            validate_editing_config(config)

    def test_memit_without_memit_config_raises_error(self) -> None:
        """Test that MEMIT without memit_config raises ValueError."""
        config = create_editing_config(method=EditingMethod.MEMIT, memit_config=None)
        with pytest.raises(ValueError, match="memit_config is required"):
            validate_editing_config(config)

    def test_other_methods_without_config_is_valid(self) -> None:
        """Test that other methods without specific config is valid."""
        config = create_editing_config(method=EditingMethod.MEND)
        validate_editing_config(config)  # Should not raise


class TestListEditingMethods:
    """Tests for list_editing_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_editing_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_editing_methods()
        assert "rome" in methods
        assert "memit" in methods
        assert "mend" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_editing_methods()
        assert methods == sorted(methods)


class TestValidateEditingMethod:
    """Tests for validate_editing_method function."""

    def test_valid_rome(self) -> None:
        """Test validation of rome."""
        assert validate_editing_method("rome") is True

    def test_valid_memit(self) -> None:
        """Test validation of memit."""
        assert validate_editing_method("memit") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_editing_method("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_editing_method("") is False


class TestGetEditingMethod:
    """Tests for get_editing_method function."""

    def test_get_rome(self) -> None:
        """Test getting ROME."""
        result = get_editing_method("rome")
        assert result == EditingMethod.ROME

    def test_get_memit(self) -> None:
        """Test getting MEMIT."""
        result = get_editing_method("memit")
        assert result == EditingMethod.MEMIT

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid editing method"):
            get_editing_method("invalid")


class TestListLocalizationTypes:
    """Tests for list_localization_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_localization_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_localization_types()
        assert "causal_tracing" in types
        assert "activation_patching" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_localization_types()
        assert types == sorted(types)


class TestValidateLocalizationType:
    """Tests for validate_localization_type function."""

    def test_valid_causal_tracing(self) -> None:
        """Test validation of causal_tracing."""
        assert validate_localization_type("causal_tracing") is True

    def test_valid_activation_patching(self) -> None:
        """Test validation of activation_patching."""
        assert validate_localization_type("activation_patching") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_localization_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_localization_type("") is False


class TestGetLocalizationType:
    """Tests for get_localization_type function."""

    def test_get_causal_tracing(self) -> None:
        """Test getting CAUSAL_TRACING."""
        result = get_localization_type("causal_tracing")
        assert result == LocalizationType.CAUSAL_TRACING

    def test_get_activation_patching(self) -> None:
        """Test getting ACTIVATION_PATCHING."""
        result = get_localization_type("activation_patching")
        assert result == LocalizationType.ACTIVATION_PATCHING

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid localization type"):
            get_localization_type("invalid")


class TestListEditScopes:
    """Tests for list_edit_scopes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        scopes = list_edit_scopes()
        assert isinstance(scopes, list)

    def test_contains_expected_scopes(self) -> None:
        """Test that list contains expected scopes."""
        scopes = list_edit_scopes()
        assert "single" in scopes
        assert "batch" in scopes
        assert "sequential" in scopes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        scopes = list_edit_scopes()
        assert scopes == sorted(scopes)


class TestValidateEditScope:
    """Tests for validate_edit_scope function."""

    def test_valid_single(self) -> None:
        """Test validation of single."""
        assert validate_edit_scope("single") is True

    def test_valid_batch(self) -> None:
        """Test validation of batch."""
        assert validate_edit_scope("batch") is True

    def test_invalid_scope(self) -> None:
        """Test validation of invalid scope."""
        assert validate_edit_scope("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_edit_scope("") is False


class TestGetEditScope:
    """Tests for get_edit_scope function."""

    def test_get_single(self) -> None:
        """Test getting SINGLE."""
        result = get_edit_scope("single")
        assert result == EditScope.SINGLE

    def test_get_batch(self) -> None:
        """Test getting BATCH."""
        result = get_edit_scope("batch")
        assert result == EditScope.BATCH

    def test_invalid_scope_raises_error(self) -> None:
        """Test that invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="invalid edit scope"):
            get_edit_scope("invalid")


class TestLocalizeKnowledge:
    """Tests for localize_knowledge function."""

    def test_basic_localization(self) -> None:
        """Test basic knowledge localization."""
        activations = [[0.1, 0.2], [0.3, 0.8], [0.9, 0.7], [0.4, 0.3]]
        clean = [0.8, 0.2]
        corrupted = [0.3, 0.7]
        result = localize_knowledge(activations, clean, corrupted)
        assert len(result.layer_scores) == 4
        assert len(result.critical_layers) > 0
        assert result.method == LocalizationType.CAUSAL_TRACING

    def test_with_activation_patching_method(self) -> None:
        """Test with activation patching method."""
        activations = [[0.5], [0.8]]
        clean = [0.9]
        corrupted = [0.1]
        result = localize_knowledge(
            activations, clean, corrupted, method=LocalizationType.ACTIVATION_PATCHING
        )
        assert result.method == LocalizationType.ACTIVATION_PATCHING

    def test_token_importance_sum(self) -> None:
        """Test that token importance sums approximately to 1."""
        activations = [[0.2, 0.3, 0.5]]
        clean = [0.7]
        corrupted = [0.2]
        result = localize_knowledge(activations, clean, corrupted)
        assert sum(result.token_importance) == pytest.approx(1.0)

    def test_none_activations_raises_error(self) -> None:
        """Test that None activations raises ValueError."""
        with pytest.raises(ValueError, match="layer_activations cannot be None"):
            localize_knowledge(None, [0.5], [0.5])  # type: ignore[arg-type]

    def test_empty_activations_raises_error(self) -> None:
        """Test that empty activations raises ValueError."""
        with pytest.raises(ValueError, match="layer_activations cannot be empty"):
            localize_knowledge([], [0.5], [0.5])

    def test_none_clean_probs_raises_error(self) -> None:
        """Test that None clean_probs raises ValueError."""
        with pytest.raises(ValueError, match="clean_probs cannot be None"):
            localize_knowledge([[0.1]], None, [0.5])  # type: ignore[arg-type]

    def test_empty_clean_probs_raises_error(self) -> None:
        """Test that empty clean_probs raises ValueError."""
        with pytest.raises(ValueError, match="clean_probs cannot be empty"):
            localize_knowledge([[0.1]], [], [0.5])

    def test_none_corrupted_probs_raises_error(self) -> None:
        """Test that None corrupted_probs raises ValueError."""
        with pytest.raises(ValueError, match="corrupted_probs cannot be None"):
            localize_knowledge([[0.1]], [0.5], None)  # type: ignore[arg-type]

    def test_empty_corrupted_probs_raises_error(self) -> None:
        """Test that empty corrupted_probs raises ValueError."""
        with pytest.raises(ValueError, match="corrupted_probs cannot be empty"):
            localize_knowledge([[0.1]], [0.5], [])

    def test_metadata_contains_num_layers(self) -> None:
        """Test that metadata contains num_layers."""
        activations = [[0.1], [0.2], [0.3]]
        clean = [0.8]
        corrupted = [0.3]
        result = localize_knowledge(activations, clean, corrupted)
        assert result.metadata is not None
        assert result.metadata["num_layers"] == 3

    def test_zero_activations_uniform_importance(self) -> None:
        """Test that zero activations result in uniform token importance."""
        activations = [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
        clean = [0.8]
        corrupted = [0.3]
        result = localize_knowledge(activations, clean, corrupted)
        # When total activation is 0, should get uniform distribution
        expected_importance = 1.0 / 3
        for importance in result.token_importance:
            assert importance == pytest.approx(expected_importance)


class TestCalculateEditSuccess:
    """Tests for calculate_edit_success function."""

    def test_all_match(self) -> None:
        """Test with all predictions matching targets."""
        predictions = ["A", "B", "C"]
        targets = ["A", "B", "C"]
        assert calculate_edit_success(predictions, targets) == pytest.approx(1.0)

    def test_none_match(self) -> None:
        """Test with no predictions matching targets."""
        predictions = ["X", "Y", "Z"]
        targets = ["A", "B", "C"]
        assert calculate_edit_success(predictions, targets) == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        """Test with partial matches."""
        predictions = ["A", "X", "C"]
        targets = ["A", "B", "C"]
        assert calculate_edit_success(predictions, targets) == pytest.approx(2 / 3)

    def test_case_insensitive(self) -> None:
        """Test that comparison is case insensitive."""
        predictions = ["ROME", "berlin"]
        targets = ["rome", "BERLIN"]
        assert calculate_edit_success(predictions, targets) == pytest.approx(1.0)

    def test_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        predictions = ["  A  ", "B"]
        targets = ["A", "  B  "]
        assert calculate_edit_success(predictions, targets) == pytest.approx(1.0)

    def test_none_predictions_raises_error(self) -> None:
        """Test that None predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be None"):
            calculate_edit_success(None, ["a"])  # type: ignore[arg-type]

    def test_empty_predictions_raises_error(self) -> None:
        """Test that empty predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            calculate_edit_success([], ["a"])

    def test_none_targets_raises_error(self) -> None:
        """Test that None targets raises ValueError."""
        with pytest.raises(ValueError, match="targets cannot be None"):
            calculate_edit_success(["a"], None)  # type: ignore[arg-type]

    def test_empty_targets_raises_error(self) -> None:
        """Test that empty targets raises ValueError."""
        with pytest.raises(ValueError, match="targets cannot be empty"):
            calculate_edit_success(["a"], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            calculate_edit_success(["a"], ["b", "c"])


class TestMeasureSpecificity:
    """Tests for measure_specificity function."""

    def test_all_preserved(self) -> None:
        """Test with all facts preserved."""
        predictions = ["Paris", "Berlin", "Tokyo"]
        ground_truths = ["Paris", "Berlin", "Tokyo"]
        assert measure_specificity(predictions, ground_truths) == pytest.approx(1.0)

    def test_none_preserved(self) -> None:
        """Test with no facts preserved."""
        predictions = ["X", "Y", "Z"]
        ground_truths = ["Paris", "Berlin", "Tokyo"]
        assert measure_specificity(predictions, ground_truths) == pytest.approx(0.0)

    def test_partial_preserved(self) -> None:
        """Test with partial preservation."""
        predictions = ["Paris", "X", "Tokyo"]
        ground_truths = ["Paris", "Berlin", "Tokyo"]
        assert measure_specificity(predictions, ground_truths) == pytest.approx(2 / 3)

    def test_none_predictions_raises_error(self) -> None:
        """Test that None predictions raises ValueError."""
        with pytest.raises(ValueError, match="unrelated_predictions cannot be None"):
            measure_specificity(None, ["a"])  # type: ignore[arg-type]

    def test_empty_predictions_raises_error(self) -> None:
        """Test that empty predictions raises ValueError."""
        with pytest.raises(ValueError, match="unrelated_predictions cannot be empty"):
            measure_specificity([], ["a"])

    def test_none_ground_truths_raises_error(self) -> None:
        """Test that None ground_truths raises ValueError."""
        with pytest.raises(ValueError, match="unrelated_ground_truths cannot be None"):
            measure_specificity(["a"], None)  # type: ignore[arg-type]

    def test_empty_ground_truths_raises_error(self) -> None:
        """Test that empty ground_truths raises ValueError."""
        with pytest.raises(ValueError, match="unrelated_ground_truths cannot be empty"):
            measure_specificity(["a"], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            measure_specificity(["a"], ["b", "c"])


class TestMeasureGeneralization:
    """Tests for measure_generalization function."""

    def test_full_generalization(self) -> None:
        """Test with full generalization."""
        predictions = ["Rome", "Rome", "Rome"]
        targets = ["Rome", "Rome", "Rome"]
        assert measure_generalization(predictions, targets) == pytest.approx(1.0)

    def test_no_generalization(self) -> None:
        """Test with no generalization."""
        predictions = ["Paris", "Paris", "Paris"]
        targets = ["Rome", "Rome", "Rome"]
        assert measure_generalization(predictions, targets) == pytest.approx(0.0)

    def test_partial_generalization(self) -> None:
        """Test with partial generalization."""
        predictions = ["Rome", "Paris", "Rome"]
        targets = ["Rome", "Rome", "Rome"]
        assert measure_generalization(predictions, targets) == pytest.approx(2 / 3)

    def test_none_predictions_raises_error(self) -> None:
        """Test that None predictions raises ValueError."""
        with pytest.raises(ValueError, match="paraphrase_predictions cannot be None"):
            measure_generalization(None, ["a"])  # type: ignore[arg-type]

    def test_empty_predictions_raises_error(self) -> None:
        """Test that empty predictions raises ValueError."""
        with pytest.raises(ValueError, match="paraphrase_predictions cannot be empty"):
            measure_generalization([], ["a"])

    def test_none_targets_raises_error(self) -> None:
        """Test that None targets raises ValueError."""
        with pytest.raises(ValueError, match="targets cannot be None"):
            measure_generalization(["a"], None)  # type: ignore[arg-type]

    def test_empty_targets_raises_error(self) -> None:
        """Test that empty targets raises ValueError."""
        with pytest.raises(ValueError, match="targets cannot be empty"):
            measure_generalization(["a"], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            measure_generalization(["a"], ["b", "c"])


class TestFormatEditingStats:
    """Tests for format_editing_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = EditingStats(
            edit_success_rate=0.95,
            specificity=0.92,
            generalization=0.85,
            locality_score=0.88,
        )
        formatted = format_editing_stats(stats)
        assert "Model Editing Evaluation Results" in formatted
        assert "Edit Success Rate: 95.00%" in formatted
        assert "Specificity: 92.00%" in formatted
        assert "Generalization: 85.00%" in formatted
        assert "Locality Score: 88.00%" in formatted

    def test_excellent_assessment(self) -> None:
        """Test excellent assessment."""
        stats = EditingStats(0.95, 0.95, 0.95, 0.95)
        formatted = format_editing_stats(stats)
        assert "Excellent" in formatted

    def test_good_assessment(self) -> None:
        """Test good assessment."""
        stats = EditingStats(0.85, 0.85, 0.85, 0.85)
        formatted = format_editing_stats(stats)
        assert "Good" in formatted

    def test_fair_assessment(self) -> None:
        """Test fair assessment."""
        stats = EditingStats(0.75, 0.75, 0.75, 0.75)
        formatted = format_editing_stats(stats)
        assert "Fair" in formatted

    def test_needs_improvement_assessment(self) -> None:
        """Test needs improvement assessment."""
        stats = EditingStats(0.5, 0.5, 0.5, 0.5)
        formatted = format_editing_stats(stats)
        assert "Needs Improvement" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_editing_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedEditingConfig:
    """Tests for get_recommended_editing_config function."""

    def test_single_edit_uses_rome(self) -> None:
        """Test that single edit uses ROME."""
        config = get_recommended_editing_config(1)
        assert config.method == EditingMethod.ROME
        assert config.rome_config is not None
        assert config.scope == EditScope.SINGLE

    def test_few_edits_uses_memit(self) -> None:
        """Test that few edits uses MEMIT."""
        config = get_recommended_editing_config(5)
        assert config.method == EditingMethod.MEMIT
        assert config.memit_config is not None
        assert config.scope == EditScope.BATCH

    def test_many_edits_uses_memit_adjusted(self) -> None:
        """Test that many edits uses MEMIT with adjusted params."""
        config = get_recommended_editing_config(100)
        assert config.method == EditingMethod.MEMIT
        assert config.memit_config is not None
        assert config.memit_config.lambda_weight == pytest.approx(10000.0)
        assert config.memit_config.edit_weight == pytest.approx(0.5)

    def test_small_model(self) -> None:
        """Test config for small model."""
        config = get_recommended_editing_config(1, model_size="small")
        assert config.rome_config is not None
        assert config.rome_config.layers == [3, 4, 5]

    def test_base_model(self) -> None:
        """Test config for base model."""
        config = get_recommended_editing_config(1, model_size="base")
        assert config.rome_config is not None
        assert config.rome_config.layers == [5, 6, 7]

    def test_large_model(self) -> None:
        """Test config for large model."""
        config = get_recommended_editing_config(1, model_size="large")
        assert config.rome_config is not None
        assert len(config.rome_config.layers) > 3

    def test_xl_model(self) -> None:
        """Test config for XL model."""
        config = get_recommended_editing_config(1, model_size="xl")
        assert config.rome_config is not None
        assert len(config.rome_config.layers) >= 5

    def test_unknown_model_uses_base(self) -> None:
        """Test that unknown model size uses base defaults."""
        config = get_recommended_editing_config(1, model_size="unknown")
        assert config.rome_config is not None
        assert config.rome_config.layers == [5, 6, 7]

    def test_zero_edits_raises_error(self) -> None:
        """Test that zero edits raises ValueError."""
        with pytest.raises(ValueError, match="num_edits must be positive"):
            get_recommended_editing_config(0)

    def test_negative_edits_raises_error(self) -> None:
        """Test that negative edits raises ValueError."""
        with pytest.raises(ValueError, match="num_edits must be positive"):
            get_recommended_editing_config(-5)

    def test_empty_model_size_raises_error(self) -> None:
        """Test that empty model_size raises ValueError."""
        with pytest.raises(ValueError, match="model_size cannot be empty"):
            get_recommended_editing_config(1, model_size="")

    def test_verify_edit_is_true(self) -> None:
        """Test that verify_edit is always True."""
        config = get_recommended_editing_config(1)
        assert config.verify_edit is True


class TestPropertyBased:
    """Property-based tests for editing functions."""

    @given(
        st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_edit_success_with_identical_inputs(self, items: list[str]) -> None:
        """Test that identical inputs give 100% success."""
        # Filter empty strings
        items = [s for s in items if s.strip()]
        if not items:
            return
        success = calculate_edit_success(items, items)
        assert success == pytest.approx(1.0)

    @given(
        st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_specificity_with_identical_inputs(self, items: list[str]) -> None:
        """Test that identical inputs give 100% specificity."""
        # Filter empty strings
        items = [s for s in items if s.strip()]
        if not items:
            return
        specificity = measure_specificity(items, items)
        assert specificity == pytest.approx(1.0)

    @given(
        st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_generalization_with_identical_inputs(self, items: list[str]) -> None:
        """Test that identical inputs give 100% generalization."""
        # Filter empty strings
        items = [s for s in items if s.strip()]
        if not items:
            return
        generalization = measure_generalization(items, items)
        assert generalization == pytest.approx(1.0)

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=10)
    def test_recommended_config_always_valid(self, num_edits: int) -> None:
        """Test that recommended config is always valid."""
        config = get_recommended_editing_config(num_edits)
        validate_editing_config(config)  # Should not raise

    @given(
        st.lists(
            st.lists(
                st.floats(
                    min_value=0.01,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=1,
                max_size=5,
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=10)
    def test_localize_knowledge_always_finds_critical_layers(
        self, activations: list[list[float]]
    ) -> None:
        """Test that localize_knowledge always finds critical layers."""
        # Ensure consistent activation sizes
        min_size = min(len(a) for a in activations)
        activations = [a[:min_size] for a in activations]
        clean = [0.8] * min_size
        corrupted = [0.3] * min_size
        result = localize_knowledge(activations, clean, corrupted)
        assert len(result.critical_layers) >= 1
