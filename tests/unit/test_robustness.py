"""Tests for robustness testing functionality."""

from __future__ import annotations

import pytest

from hf_gtc.evaluation.robustness import (
    AdversarialConfig,
    AttackMethod,
    OODConfig,
    OODDetectionMethod,
    PerturbationConfig,
    PerturbationType,
    RobustnessResult,
    apply_perturbation,
    calculate_attack_success_rate,
    calculate_robustness_score,
    create_adversarial_config,
    create_ood_config,
    create_perturbation_config,
    detect_ood_samples,
    format_robustness_result,
    get_attack_method,
    get_ood_detection_method,
    get_perturbation_type,
    get_recommended_robustness_config,
    list_attack_methods,
    list_ood_detection_methods,
    list_perturbation_types,
    validate_adversarial_config,
    validate_ood_config,
    validate_perturbation_config,
)


class TestPerturbationType:
    """Tests for PerturbationType enum."""

    def test_typo_value(self) -> None:
        """Test TYPO value."""
        assert PerturbationType.TYPO.value == "typo"

    def test_synonym_value(self) -> None:
        """Test SYNONYM value."""
        assert PerturbationType.SYNONYM.value == "synonym"

    def test_deletion_value(self) -> None:
        """Test DELETION value."""
        assert PerturbationType.DELETION.value == "deletion"

    def test_insertion_value(self) -> None:
        """Test INSERTION value."""
        assert PerturbationType.INSERTION.value == "insertion"

    def test_paraphrase_value(self) -> None:
        """Test PARAPHRASE value."""
        assert PerturbationType.PARAPHRASE.value == "paraphrase"


class TestAttackMethod:
    """Tests for AttackMethod enum."""

    def test_textfooler_value(self) -> None:
        """Test TEXTFOOLER value."""
        assert AttackMethod.TEXTFOOLER.value == "textfooler"

    def test_bert_attack_value(self) -> None:
        """Test BERT_ATTACK value."""
        assert AttackMethod.BERT_ATTACK.value == "bert_attack"

    def test_deepwordbug_value(self) -> None:
        """Test DEEPWORDBUG value."""
        assert AttackMethod.DEEPWORDBUG.value == "deepwordbug"

    def test_pwws_value(self) -> None:
        """Test PWWS value."""
        assert AttackMethod.PWWS.value == "pwws"


class TestOODDetectionMethod:
    """Tests for OODDetectionMethod enum."""

    def test_mahalanobis_value(self) -> None:
        """Test MAHALANOBIS value."""
        assert OODDetectionMethod.MAHALANOBIS.value == "mahalanobis"

    def test_energy_value(self) -> None:
        """Test ENERGY value."""
        assert OODDetectionMethod.ENERGY.value == "energy"

    def test_entropy_value(self) -> None:
        """Test ENTROPY value."""
        assert OODDetectionMethod.ENTROPY.value == "entropy"

    def test_msp_value(self) -> None:
        """Test MSP value."""
        assert OODDetectionMethod.MSP.value == "msp"


class TestPerturbationConfig:
    """Tests for PerturbationConfig dataclass."""

    def test_required_perturbation_type(self) -> None:
        """Test that perturbation_type is required."""
        config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        assert config.perturbation_type == PerturbationType.TYPO

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        assert config.intensity == pytest.approx(0.1)
        assert config.max_perturbations == 5
        assert config.preserve_semantics is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.SYNONYM,
            intensity=0.3,
            max_perturbations=10,
            preserve_semantics=False,
        )
        assert config.perturbation_type == PerturbationType.SYNONYM
        assert config.intensity == pytest.approx(0.3)
        assert config.max_perturbations == 10
        assert config.preserve_semantics is False

    def test_frozen(self) -> None:
        """Test that PerturbationConfig is immutable."""
        config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        with pytest.raises(AttributeError):
            config.intensity = 0.5  # type: ignore[misc]


class TestAdversarialConfig:
    """Tests for AdversarialConfig dataclass."""

    def test_required_attack_method(self) -> None:
        """Test that attack_method is required."""
        config = AdversarialConfig(attack_method=AttackMethod.TEXTFOOLER)
        assert config.attack_method == AttackMethod.TEXTFOOLER

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AdversarialConfig(attack_method=AttackMethod.TEXTFOOLER)
        assert config.max_queries == 100
        assert config.success_threshold == pytest.approx(0.5)
        assert config.target_label is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AdversarialConfig(
            attack_method=AttackMethod.BERT_ATTACK,
            max_queries=200,
            success_threshold=0.3,
            target_label=1,
        )
        assert config.attack_method == AttackMethod.BERT_ATTACK
        assert config.max_queries == 200
        assert config.success_threshold == pytest.approx(0.3)
        assert config.target_label == 1

    def test_frozen(self) -> None:
        """Test that AdversarialConfig is immutable."""
        config = AdversarialConfig(attack_method=AttackMethod.TEXTFOOLER)
        with pytest.raises(AttributeError):
            config.max_queries = 500  # type: ignore[misc]


class TestOODConfig:
    """Tests for OODConfig dataclass."""

    def test_required_detection_method(self) -> None:
        """Test that detection_method is required."""
        config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        assert config.detection_method == OODDetectionMethod.ENERGY

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        assert config.threshold == pytest.approx(0.5)
        assert config.calibration_data_size == 1000
        assert config.temperature == pytest.approx(1.0)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = OODConfig(
            detection_method=OODDetectionMethod.MAHALANOBIS,
            threshold=0.7,
            calibration_data_size=500,
            temperature=2.0,
        )
        assert config.detection_method == OODDetectionMethod.MAHALANOBIS
        assert config.threshold == pytest.approx(0.7)
        assert config.calibration_data_size == 500
        assert config.temperature == pytest.approx(2.0)

    def test_frozen(self) -> None:
        """Test that OODConfig is immutable."""
        config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        with pytest.raises(AttributeError):
            config.threshold = 0.9  # type: ignore[misc]


class TestRobustnessResult:
    """Tests for RobustnessResult dataclass."""

    def test_creation(self) -> None:
        """Test creating RobustnessResult instance."""
        result = RobustnessResult(
            accuracy_under_perturbation=0.85,
            attack_success_rate=0.15,
            ood_detection_auroc=0.92,
        )
        assert result.accuracy_under_perturbation == pytest.approx(0.85)
        assert result.attack_success_rate == pytest.approx(0.15)
        assert result.ood_detection_auroc == pytest.approx(0.92)

    def test_frozen(self) -> None:
        """Test that RobustnessResult is immutable."""
        result = RobustnessResult(0.85, 0.15, 0.92)
        with pytest.raises(AttributeError):
            result.accuracy_under_perturbation = 0.9  # type: ignore[misc]


class TestValidatePerturbationConfig:
    """Tests for validate_perturbation_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        validate_perturbation_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_perturbation_config(None)  # type: ignore[arg-type]

    def test_intensity_too_low_raises_error(self) -> None:
        """Test that intensity < 0 raises ValueError."""
        config = PerturbationConfig(PerturbationType.TYPO, intensity=-0.1)
        with pytest.raises(ValueError, match="intensity must be between 0 and 1"):
            validate_perturbation_config(config)

    def test_intensity_too_high_raises_error(self) -> None:
        """Test that intensity > 1 raises ValueError."""
        config = PerturbationConfig(PerturbationType.TYPO, intensity=1.5)
        with pytest.raises(ValueError, match="intensity must be between 0 and 1"):
            validate_perturbation_config(config)

    def test_non_positive_max_perturbations_raises_error(self) -> None:
        """Test that non-positive max_perturbations raises ValueError."""
        config = PerturbationConfig(PerturbationType.TYPO, max_perturbations=0)
        with pytest.raises(ValueError, match="max_perturbations must be positive"):
            validate_perturbation_config(config)


class TestValidateAdversarialConfig:
    """Tests for validate_adversarial_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = AdversarialConfig(attack_method=AttackMethod.TEXTFOOLER)
        validate_adversarial_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_adversarial_config(None)  # type: ignore[arg-type]

    def test_non_positive_max_queries_raises_error(self) -> None:
        """Test that non-positive max_queries raises ValueError."""
        config = AdversarialConfig(AttackMethod.TEXTFOOLER, max_queries=0)
        with pytest.raises(ValueError, match="max_queries must be positive"):
            validate_adversarial_config(config)

    def test_success_threshold_too_low_raises_error(self) -> None:
        """Test that success_threshold < 0 raises ValueError."""
        config = AdversarialConfig(AttackMethod.TEXTFOOLER, success_threshold=-0.1)
        with pytest.raises(ValueError, match="success_threshold must be between"):
            validate_adversarial_config(config)

    def test_success_threshold_too_high_raises_error(self) -> None:
        """Test that success_threshold > 1 raises ValueError."""
        config = AdversarialConfig(AttackMethod.TEXTFOOLER, success_threshold=1.5)
        with pytest.raises(ValueError, match="success_threshold must be between"):
            validate_adversarial_config(config)


class TestValidateOODConfig:
    """Tests for validate_ood_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        validate_ood_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_ood_config(None)  # type: ignore[arg-type]

    def test_threshold_too_low_raises_error(self) -> None:
        """Test that threshold < 0 raises ValueError."""
        config = OODConfig(OODDetectionMethod.ENERGY, threshold=-0.1)
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            validate_ood_config(config)

    def test_threshold_too_high_raises_error(self) -> None:
        """Test that threshold > 1 raises ValueError."""
        config = OODConfig(OODDetectionMethod.ENERGY, threshold=1.5)
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            validate_ood_config(config)

    def test_non_positive_calibration_data_size_raises_error(self) -> None:
        """Test that non-positive calibration_data_size raises ValueError."""
        config = OODConfig(OODDetectionMethod.ENERGY, calibration_data_size=0)
        with pytest.raises(ValueError, match="calibration_data_size must be positive"):
            validate_ood_config(config)

    def test_non_positive_temperature_raises_error(self) -> None:
        """Test that non-positive temperature raises ValueError."""
        config = OODConfig(OODDetectionMethod.ENERGY, temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_ood_config(config)


class TestCreatePerturbationConfig:
    """Tests for create_perturbation_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_perturbation_config(PerturbationType.TYPO)
        assert isinstance(config, PerturbationConfig)
        assert config.perturbation_type == PerturbationType.TYPO

    def test_custom_values(self) -> None:
        """Test with custom values."""
        config = create_perturbation_config(
            PerturbationType.SYNONYM,
            intensity=0.3,
            max_perturbations=10,
            preserve_semantics=False,
        )
        assert config.intensity == pytest.approx(0.3)
        assert config.max_perturbations == 10
        assert config.preserve_semantics is False

    def test_invalid_intensity_raises_error(self) -> None:
        """Test that invalid intensity raises ValueError."""
        with pytest.raises(ValueError, match="intensity"):
            create_perturbation_config(PerturbationType.TYPO, intensity=2.0)


class TestCreateAdversarialConfig:
    """Tests for create_adversarial_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_adversarial_config(AttackMethod.TEXTFOOLER)
        assert isinstance(config, AdversarialConfig)
        assert config.attack_method == AttackMethod.TEXTFOOLER

    def test_custom_values(self) -> None:
        """Test with custom values."""
        config = create_adversarial_config(
            AttackMethod.BERT_ATTACK,
            max_queries=200,
            success_threshold=0.3,
            target_label=1,
        )
        assert config.max_queries == 200
        assert config.success_threshold == pytest.approx(0.3)
        assert config.target_label == 1

    def test_invalid_max_queries_raises_error(self) -> None:
        """Test that invalid max_queries raises ValueError."""
        with pytest.raises(ValueError, match="max_queries"):
            create_adversarial_config(AttackMethod.TEXTFOOLER, max_queries=-1)


class TestCreateOODConfig:
    """Tests for create_ood_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_ood_config(OODDetectionMethod.ENERGY)
        assert isinstance(config, OODConfig)
        assert config.detection_method == OODDetectionMethod.ENERGY

    def test_custom_values(self) -> None:
        """Test with custom values."""
        config = create_ood_config(
            OODDetectionMethod.MAHALANOBIS,
            threshold=0.7,
            calibration_data_size=500,
            temperature=2.0,
        )
        assert config.threshold == pytest.approx(0.7)
        assert config.calibration_data_size == 500
        assert config.temperature == pytest.approx(2.0)

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            create_ood_config(OODDetectionMethod.ENERGY, threshold=1.5)


class TestListPerturbationTypes:
    """Tests for list_perturbation_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_perturbation_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_perturbation_types()
        assert "typo" in types
        assert "synonym" in types
        assert "deletion" in types
        assert "insertion" in types
        assert "paraphrase" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_perturbation_types()
        assert types == sorted(types)


class TestGetPerturbationType:
    """Tests for get_perturbation_type function."""

    def test_get_typo(self) -> None:
        """Test getting TYPO type."""
        result = get_perturbation_type("typo")
        assert result == PerturbationType.TYPO

    def test_get_synonym(self) -> None:
        """Test getting SYNONYM type."""
        result = get_perturbation_type("synonym")
        assert result == PerturbationType.SYNONYM

    def test_get_deletion(self) -> None:
        """Test getting DELETION type."""
        result = get_perturbation_type("deletion")
        assert result == PerturbationType.DELETION

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid perturbation type"):
            get_perturbation_type("invalid")


class TestListAttackMethods:
    """Tests for list_attack_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_attack_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_attack_methods()
        assert "textfooler" in methods
        assert "bert_attack" in methods
        assert "deepwordbug" in methods
        assert "pwws" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_attack_methods()
        assert methods == sorted(methods)


class TestGetAttackMethod:
    """Tests for get_attack_method function."""

    def test_get_textfooler(self) -> None:
        """Test getting TEXTFOOLER method."""
        result = get_attack_method("textfooler")
        assert result == AttackMethod.TEXTFOOLER

    def test_get_bert_attack(self) -> None:
        """Test getting BERT_ATTACK method."""
        result = get_attack_method("bert_attack")
        assert result == AttackMethod.BERT_ATTACK

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid attack method"):
            get_attack_method("invalid")


class TestListOODDetectionMethods:
    """Tests for list_ood_detection_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_ood_detection_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_ood_detection_methods()
        assert "mahalanobis" in methods
        assert "energy" in methods
        assert "entropy" in methods
        assert "msp" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_ood_detection_methods()
        assert methods == sorted(methods)


class TestGetOODDetectionMethod:
    """Tests for get_ood_detection_method function."""

    def test_get_energy(self) -> None:
        """Test getting ENERGY method."""
        result = get_ood_detection_method("energy")
        assert result == OODDetectionMethod.ENERGY

    def test_get_mahalanobis(self) -> None:
        """Test getting MAHALANOBIS method."""
        result = get_ood_detection_method("mahalanobis")
        assert result == OODDetectionMethod.MAHALANOBIS

    def test_get_entropy(self) -> None:
        """Test getting ENTROPY method."""
        result = get_ood_detection_method("entropy")
        assert result == OODDetectionMethod.ENTROPY

    def test_get_msp(self) -> None:
        """Test getting MSP method."""
        result = get_ood_detection_method("msp")
        assert result == OODDetectionMethod.MSP

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid OOD detection method"):
            get_ood_detection_method("invalid")


class TestApplyPerturbation:
    """Tests for apply_perturbation function."""

    def test_typo_perturbation(self) -> None:
        """Test applying typo perturbation."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.TYPO,
            intensity=0.5,
            max_perturbations=2,
        )
        result = apply_perturbation("hello world", config)
        assert isinstance(result, str)
        assert len(result) == len("hello world")

    def test_synonym_perturbation(self) -> None:
        """Test applying synonym perturbation."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.SYNONYM,
            intensity=0.5,
            max_perturbations=2,
        )
        result = apply_perturbation("This is a good day", config)
        assert isinstance(result, str)
        # "good" should be replaced with "excellent"
        assert "excellent" in result or "good" in result

    def test_deletion_perturbation(self) -> None:
        """Test applying deletion perturbation."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.DELETION,
            intensity=0.5,
            max_perturbations=1,
        )
        result = apply_perturbation("hello world test", config)
        assert isinstance(result, str)
        # Should have fewer words
        assert len(result.split()) < len(["hello", "world", "test"])

    def test_insertion_perturbation(self) -> None:
        """Test applying insertion perturbation."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.INSERTION,
            intensity=0.5,
            max_perturbations=1,
        )
        result = apply_perturbation("hello world", config)
        assert isinstance(result, str)
        # Should have more words
        assert len(result.split()) > len(["hello", "world"])

    def test_paraphrase_perturbation(self) -> None:
        """Test applying paraphrase perturbation."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.PARAPHRASE,
            intensity=0.5,
            max_perturbations=1,
        )
        result = apply_perturbation("This is a test", config)
        assert isinstance(result, str)

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        with pytest.raises(ValueError, match="text cannot be None"):
            apply_perturbation(None, config)  # type: ignore[arg-type]

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises ValueError."""
        config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        with pytest.raises(ValueError, match="text cannot be empty"):
            apply_perturbation("", config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            apply_perturbation("hello", None)  # type: ignore[arg-type]

    def test_short_text_typo(self) -> None:
        """Test typo perturbation on short text."""
        config = PerturbationConfig(perturbation_type=PerturbationType.TYPO)
        result = apply_perturbation("a", config)
        assert result == "a"  # Too short to perturb

    def test_single_word_deletion(self) -> None:
        """Test deletion on single word."""
        config = PerturbationConfig(perturbation_type=PerturbationType.DELETION)
        result = apply_perturbation("hello", config)
        assert result == "hello"  # Can't delete the only word


class TestCalculateRobustnessScore:
    """Tests for calculate_robustness_score function."""

    def test_perfect_robustness(self) -> None:
        """Test score when perturbed equals original."""
        score = calculate_robustness_score(0.9, 0.9)
        assert score == pytest.approx(1.0)

    def test_partial_degradation(self) -> None:
        """Test score with partial degradation."""
        score = calculate_robustness_score(0.9, 0.81)
        assert score == pytest.approx(0.9)

    def test_zero_original_accuracy(self) -> None:
        """Test with zero original accuracy."""
        score = calculate_robustness_score(0.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_improvement_capped_at_one(self) -> None:
        """Test that improvement is capped at 1.0."""
        score = calculate_robustness_score(0.5, 0.6)
        assert score == pytest.approx(1.0)

    def test_invalid_original_accuracy_raises_error(self) -> None:
        """Test that invalid original_accuracy raises ValueError."""
        with pytest.raises(ValueError, match="original_accuracy must be between"):
            calculate_robustness_score(1.5, 0.5)

    def test_invalid_perturbed_accuracy_raises_error(self) -> None:
        """Test that invalid perturbed_accuracy raises ValueError."""
        with pytest.raises(ValueError, match="perturbed_accuracy must be between"):
            calculate_robustness_score(0.9, -0.1)

    def test_negative_original_accuracy_raises_error(self) -> None:
        """Test that negative original_accuracy raises ValueError."""
        with pytest.raises(ValueError, match="original_accuracy must be between"):
            calculate_robustness_score(-0.1, 0.5)

    def test_zero_original_nonzero_perturbed(self) -> None:
        """Test edge case: zero original, nonzero perturbed."""
        score = calculate_robustness_score(0.0, 0.5)
        assert score == pytest.approx(1.0)


class TestDetectOODSamples:
    """Tests for detect_ood_samples function."""

    def test_basic_detection(self) -> None:
        """Test basic OOD detection."""
        config = OODConfig(
            detection_method=OODDetectionMethod.ENERGY,
            threshold=0.5,
        )
        scores = [0.3, 0.6, 0.4, 0.8]
        result = detect_ood_samples(scores, config)
        assert result == [False, True, False, True]

    def test_all_ood(self) -> None:
        """Test when all samples are OOD."""
        config = OODConfig(
            detection_method=OODDetectionMethod.ENERGY,
            threshold=0.1,
        )
        scores = [0.5, 0.6, 0.7]
        result = detect_ood_samples(scores, config)
        assert result == [True, True, True]

    def test_none_ood(self) -> None:
        """Test when no samples are OOD."""
        config = OODConfig(
            detection_method=OODDetectionMethod.ENERGY,
            threshold=0.9,
        )
        scores = [0.5, 0.6, 0.7]
        result = detect_ood_samples(scores, config)
        assert result == [False, False, False]

    def test_temperature_scaling(self) -> None:
        """Test temperature scaling effect."""
        config = OODConfig(
            detection_method=OODDetectionMethod.ENERGY,
            threshold=0.5,
            temperature=2.0,
        )
        # With temperature=2.0, scores are halved before comparison
        scores = [0.8, 1.2]  # Scaled: 0.4, 0.6
        result = detect_ood_samples(scores, config)
        assert result == [False, True]

    def test_msp_no_temperature_scaling(self) -> None:
        """Test MSP method doesn't apply temperature scaling."""
        config = OODConfig(
            detection_method=OODDetectionMethod.MSP,
            threshold=0.5,
            temperature=2.0,
        )
        scores = [0.3, 0.7]
        result = detect_ood_samples(scores, config)
        # MSP should NOT scale by temperature
        assert result == [False, True]

    def test_none_scores_raises_error(self) -> None:
        """Test that None scores raises ValueError."""
        config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        with pytest.raises(ValueError, match="scores cannot be None"):
            detect_ood_samples(None, config)  # type: ignore[arg-type]

    def test_empty_scores_raises_error(self) -> None:
        """Test that empty scores raises ValueError."""
        config = OODConfig(detection_method=OODDetectionMethod.ENERGY)
        with pytest.raises(ValueError, match="scores cannot be empty"):
            detect_ood_samples([], config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            detect_ood_samples([0.5], None)  # type: ignore[arg-type]


class TestCalculateAttackSuccessRate:
    """Tests for calculate_attack_success_rate function."""

    def test_basic_calculation(self) -> None:
        """Test basic attack success rate calculation."""
        original = [1, 0, 1, 1]
        attacked = [0, 0, 0, 1]
        labels = [1, 0, 1, 1]
        rate = calculate_attack_success_rate(original, attacked, labels)
        # 4 originally correct ([0]=1=1, [1]=0=0, [2]=1=1, [3]=1=1)
        # 2 became incorrect ([0]:1->0, [2]:1->0)
        # Rate = 2/4 = 0.5
        assert rate == pytest.approx(0.5)

    def test_no_successful_attacks(self) -> None:
        """Test when no attacks succeed."""
        original = [1, 1, 1]
        attacked = [1, 1, 1]
        labels = [1, 1, 1]
        rate = calculate_attack_success_rate(original, attacked, labels)
        assert rate == pytest.approx(0.0)

    def test_all_attacks_succeed(self) -> None:
        """Test when all attacks succeed."""
        original = [1, 1]
        attacked = [0, 0]
        labels = [1, 1]
        rate = calculate_attack_success_rate(original, attacked, labels)
        assert rate == pytest.approx(1.0)

    def test_no_originally_correct(self) -> None:
        """Test when no predictions were originally correct."""
        original = [0, 0, 0]
        attacked = [1, 1, 1]
        labels = [1, 1, 1]
        rate = calculate_attack_success_rate(original, attacked, labels)
        assert rate == pytest.approx(0.0)

    def test_none_original_raises_error(self) -> None:
        """Test that None original_predictions raises ValueError."""
        with pytest.raises(ValueError, match="original_predictions cannot be None"):
            calculate_attack_success_rate(None, [1], [1])  # type: ignore[arg-type]

    def test_none_attacked_raises_error(self) -> None:
        """Test that None attacked_predictions raises ValueError."""
        with pytest.raises(ValueError, match="attacked_predictions cannot be None"):
            calculate_attack_success_rate([1], None, [1])  # type: ignore[arg-type]

    def test_none_labels_raises_error(self) -> None:
        """Test that None original_labels raises ValueError."""
        with pytest.raises(ValueError, match="original_labels cannot be None"):
            calculate_attack_success_rate([1], [1], None)  # type: ignore[arg-type]

    def test_empty_inputs_raises_error(self) -> None:
        """Test that empty inputs raises ValueError."""
        with pytest.raises(ValueError, match="inputs cannot be empty"):
            calculate_attack_success_rate([], [], [])

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="inputs must have same length"):
            calculate_attack_success_rate([1, 1], [1], [1, 1])


class TestFormatRobustnessResult:
    """Tests for format_robustness_result function."""

    def test_format_good_results(self) -> None:
        """Test formatting good robustness results."""
        result = RobustnessResult(
            accuracy_under_perturbation=0.85,
            attack_success_rate=0.15,
            ood_detection_auroc=0.92,
        )
        formatted = format_robustness_result(result)
        assert "85.00%" in formatted
        assert "15.00%" in formatted
        assert "0.9200" in formatted
        assert "Good perturbation robustness" in formatted
        assert "Good adversarial robustness" in formatted
        assert "Excellent OOD detection" in formatted

    def test_format_moderate_results(self) -> None:
        """Test formatting moderate robustness results."""
        result = RobustnessResult(
            accuracy_under_perturbation=0.65,
            attack_success_rate=0.35,
            ood_detection_auroc=0.75,
        )
        formatted = format_robustness_result(result)
        assert "Moderate perturbation robustness" in formatted
        assert "Moderate adversarial robustness" in formatted
        assert "Good OOD detection" in formatted

    def test_format_poor_results(self) -> None:
        """Test formatting poor robustness results."""
        result = RobustnessResult(
            accuracy_under_perturbation=0.45,
            attack_success_rate=0.55,
            ood_detection_auroc=0.55,
        )
        formatted = format_robustness_result(result)
        assert "Low perturbation robustness" in formatted
        assert "Low adversarial robustness" in formatted
        assert "Poor OOD detection" in formatted

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="result cannot be None"):
            format_robustness_result(None)  # type: ignore[arg-type]

    def test_format_contains_header(self) -> None:
        """Test that format contains header."""
        result = RobustnessResult(0.85, 0.15, 0.92)
        formatted = format_robustness_result(result)
        assert "Robustness Evaluation Results" in formatted


class TestGetRecommendedRobustnessConfig:
    """Tests for get_recommended_robustness_config function."""

    def test_text_classification_config(self) -> None:
        """Test config for text classification models."""
        config = get_recommended_robustness_config("text_classification")
        assert "perturbation" in config
        assert "adversarial" in config
        assert "ood" in config
        assert config["perturbation"]["type"] == PerturbationType.SYNONYM
        assert config["adversarial"]["method"] == AttackMethod.TEXTFOOLER
        assert config["ood"]["method"] == OODDetectionMethod.MSP

    def test_sentiment_config(self) -> None:
        """Test config for sentiment models."""
        config = get_recommended_robustness_config("sentiment")
        assert config["adversarial"]["method"] == AttackMethod.BERT_ATTACK
        assert config["ood"]["method"] == OODDetectionMethod.ENTROPY

    def test_nli_config(self) -> None:
        """Test config for NLI models."""
        config = get_recommended_robustness_config("nli")
        assert config["perturbation"]["type"] == PerturbationType.PARAPHRASE
        assert config["adversarial"]["method"] == AttackMethod.PWWS
        assert config["ood"]["method"] == OODDetectionMethod.MAHALANOBIS

    def test_qa_config(self) -> None:
        """Test config for QA models."""
        config = get_recommended_robustness_config("qa")
        assert config["perturbation"]["type"] == PerturbationType.TYPO
        assert config["adversarial"]["method"] == AttackMethod.DEEPWORDBUG
        assert config["ood"]["method"] == OODDetectionMethod.ENERGY

    def test_unknown_model_type(self) -> None:
        """Test config for unknown model type returns default."""
        config = get_recommended_robustness_config("unknown_model")
        assert "perturbation" in config
        assert "adversarial" in config
        assert "ood" in config

    def test_case_insensitive(self) -> None:
        """Test that model_type is case insensitive."""
        config1 = get_recommended_robustness_config("TEXT_CLASSIFICATION")
        config2 = get_recommended_robustness_config("text_classification")
        assert config1 == config2

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_robustness_config("")

    def test_none_model_type_raises_error(self) -> None:
        """Test that None model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be None"):
            get_recommended_robustness_config(None)  # type: ignore[arg-type]


class TestPerturbationTypeCoverage:
    """Additional tests for full perturbation type coverage."""

    def test_synonym_preserves_unknown_words(self) -> None:
        """Test that unknown words are preserved."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.SYNONYM,
            intensity=1.0,
            max_perturbations=10,
        )
        result = apply_perturbation("unknown xyz words here", config)
        # Should keep original words that aren't in synonym dict
        assert "unknown" in result or "xyz" in result

    def test_synonym_preserves_case(self) -> None:
        """Test that synonym preserves case."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.SYNONYM,
            intensity=1.0,
            max_perturbations=5,
        )
        # Test uppercase
        result = apply_perturbation("GOOD day", config)
        assert "EXCELLENT" in result or "GOOD" in result

        # Test capitalized
        result = apply_perturbation("Good day", config)
        assert "Excellent" in result or "Good" in result

    def test_paraphrase_with_patterns(self) -> None:
        """Test paraphrase with matching patterns."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.PARAPHRASE,
            intensity=1.0,
            max_perturbations=5,
        )
        result = apply_perturbation("This is a test and it can work", config)
        assert isinstance(result, str)

    def test_insertion_multiple(self) -> None:
        """Test multiple word insertions."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.INSERTION,
            intensity=1.0,
            max_perturbations=3,
        )
        result = apply_perturbation("hello world test example", config)
        assert len(result.split()) > len(["hello", "world", "test", "example"])


class TestEdgeCases:
    """Tests for edge cases."""

    def test_boundary_intensity_zero(self) -> None:
        """Test intensity at zero boundary."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.TYPO,
            intensity=0.0,
            max_perturbations=5,
        )
        validate_perturbation_config(config)  # Should not raise

    def test_boundary_intensity_one(self) -> None:
        """Test intensity at one boundary."""
        config = PerturbationConfig(
            perturbation_type=PerturbationType.TYPO,
            intensity=1.0,
            max_perturbations=5,
        )
        validate_perturbation_config(config)  # Should not raise

    def test_boundary_threshold_zero(self) -> None:
        """Test threshold at zero boundary."""
        config = OODConfig(
            detection_method=OODDetectionMethod.ENERGY,
            threshold=0.0,
        )
        validate_ood_config(config)  # Should not raise

    def test_boundary_threshold_one(self) -> None:
        """Test threshold at one boundary."""
        config = OODConfig(
            detection_method=OODDetectionMethod.ENERGY,
            threshold=1.0,
        )
        validate_ood_config(config)  # Should not raise

    def test_robustness_score_full_degradation(self) -> None:
        """Test robustness score with full degradation."""
        score = calculate_robustness_score(1.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_attack_success_partial_correct(self) -> None:
        """Test attack success with mixed original predictions."""
        original = [1, 0, 1, 0]  # 2 correct, 2 incorrect
        attacked = [0, 0, 1, 1]  # 1 flipped from correct, 1 stayed correct
        labels = [1, 1, 1, 1]
        rate = calculate_attack_success_rate(original, attacked, labels)
        # Only original[0] was correct and became incorrect
        # original[2] was correct and stayed correct
        assert rate == pytest.approx(0.5)
