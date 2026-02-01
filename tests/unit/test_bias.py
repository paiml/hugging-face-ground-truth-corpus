"""Tests for bias detection functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.bias import (
    BiasAuditResult,
    BiasDetectionConfig,
    BiasType,
    FairnessConstraint,
    FairnessMetric,
    MitigationStrategy,
    StereotypeConfig,
    calculate_demographic_parity,
    calculate_disparity_score,
    calculate_equalized_odds,
    create_bias_detection_config,
    create_fairness_constraint,
    create_stereotype_config,
    detect_stereotypes,
    format_bias_audit,
    get_bias_type,
    get_fairness_metric,
    get_mitigation_strategy,
    get_recommended_bias_config,
    list_bias_types,
    list_fairness_metrics,
    list_mitigation_strategies,
    validate_bias_detection_config,
    validate_bias_type,
    validate_fairness_constraint,
    validate_fairness_metric,
    validate_mitigation_strategy,
    validate_stereotype_config,
)


class TestFairnessMetric:
    """Tests for FairnessMetric enum."""

    def test_demographic_parity_value(self) -> None:
        """Test DEMOGRAPHIC_PARITY value."""
        assert FairnessMetric.DEMOGRAPHIC_PARITY.value == "demographic_parity"

    def test_equalized_odds_value(self) -> None:
        """Test EQUALIZED_ODDS value."""
        assert FairnessMetric.EQUALIZED_ODDS.value == "equalized_odds"

    def test_calibration_value(self) -> None:
        """Test CALIBRATION value."""
        assert FairnessMetric.CALIBRATION.value == "calibration"

    def test_predictive_parity_value(self) -> None:
        """Test PREDICTIVE_PARITY value."""
        assert FairnessMetric.PREDICTIVE_PARITY.value == "predictive_parity"


class TestBiasType:
    """Tests for BiasType enum."""

    def test_gender_value(self) -> None:
        """Test GENDER value."""
        assert BiasType.GENDER.value == "gender"

    def test_race_value(self) -> None:
        """Test RACE value."""
        assert BiasType.RACE.value == "race"

    def test_age_value(self) -> None:
        """Test AGE value."""
        assert BiasType.AGE.value == "age"

    def test_religion_value(self) -> None:
        """Test RELIGION value."""
        assert BiasType.RELIGION.value == "religion"

    def test_nationality_value(self) -> None:
        """Test NATIONALITY value."""
        assert BiasType.NATIONALITY.value == "nationality"


class TestMitigationStrategy:
    """Tests for MitigationStrategy enum."""

    def test_reweighting_value(self) -> None:
        """Test REWEIGHTING value."""
        assert MitigationStrategy.REWEIGHTING.value == "reweighting"

    def test_resampling_value(self) -> None:
        """Test RESAMPLING value."""
        assert MitigationStrategy.RESAMPLING.value == "resampling"

    def test_adversarial_value(self) -> None:
        """Test ADVERSARIAL value."""
        assert MitigationStrategy.ADVERSARIAL.value == "adversarial"

    def test_calibration_value(self) -> None:
        """Test CALIBRATION value."""
        assert MitigationStrategy.CALIBRATION.value == "calibration"


class TestBiasDetectionConfig:
    """Tests for BiasDetectionConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating BiasDetectionConfig instance."""
        config = BiasDetectionConfig(
            protected_attributes=["gender", "race"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        assert config.protected_attributes == ["gender", "race"]
        assert config.metrics == [FairnessMetric.DEMOGRAPHIC_PARITY]

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BiasDetectionConfig(
            protected_attributes=["gender"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        assert config.threshold == pytest.approx(0.1)
        assert config.intersectional is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BiasDetectionConfig(
            protected_attributes=["gender", "race"],
            metrics=[FairnessMetric.EQUALIZED_ODDS],
            threshold=0.05,
            intersectional=True,
        )
        assert config.threshold == pytest.approx(0.05)
        assert config.intersectional is True

    def test_frozen(self) -> None:
        """Test that BiasDetectionConfig is immutable."""
        config = BiasDetectionConfig(
            protected_attributes=["gender"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        with pytest.raises(AttributeError):
            config.threshold = 0.2  # type: ignore[misc]


class TestFairnessConstraint:
    """Tests for FairnessConstraint dataclass."""

    def test_creation(self) -> None:
        """Test creating FairnessConstraint instance."""
        constraint = FairnessConstraint(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            threshold=0.1,
            group_comparison="pairwise",
        )
        assert constraint.metric == FairnessMetric.DEMOGRAPHIC_PARITY
        assert constraint.threshold == pytest.approx(0.1)

    def test_default_values(self) -> None:
        """Test default constraint values."""
        constraint = FairnessConstraint(metric=FairnessMetric.EQUALIZED_ODDS)
        assert constraint.threshold == pytest.approx(0.05)
        assert constraint.group_comparison == "pairwise"
        assert constraint.slack == pytest.approx(0.0)

    def test_frozen(self) -> None:
        """Test that FairnessConstraint is immutable."""
        constraint = FairnessConstraint(metric=FairnessMetric.DEMOGRAPHIC_PARITY)
        with pytest.raises(AttributeError):
            constraint.slack = 0.1  # type: ignore[misc]


class TestBiasAuditResult:
    """Tests for BiasAuditResult dataclass."""

    def test_creation(self) -> None:
        """Test creating BiasAuditResult instance."""
        result = BiasAuditResult(
            disparities={"gender": 0.15},
            scores_by_group={"male": 0.8, "female": 0.65},
            overall_fairness=0.75,
            recommendations=["Consider reweighting"],
        )
        assert result.overall_fairness == pytest.approx(0.75)
        assert result.disparities["gender"] == pytest.approx(0.15)

    def test_frozen(self) -> None:
        """Test that BiasAuditResult is immutable."""
        result = BiasAuditResult(
            disparities={},
            scores_by_group={},
            overall_fairness=0.9,
            recommendations=[],
        )
        with pytest.raises(AttributeError):
            result.overall_fairness = 0.5  # type: ignore[misc]


class TestStereotypeConfig:
    """Tests for StereotypeConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating StereotypeConfig instance."""
        config = StereotypeConfig(
            lexicon_path="/path/to/lexicon",
            categories=["gender", "race"],
            sensitivity=0.7,
        )
        assert config.lexicon_path == "/path/to/lexicon"
        assert config.categories == ["gender", "race"]
        assert config.sensitivity == pytest.approx(0.7)

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = StereotypeConfig()
        assert config.lexicon_path is None
        assert config.categories == []
        assert config.sensitivity == pytest.approx(0.5)
        assert config.context_window == 5

    def test_frozen(self) -> None:
        """Test that StereotypeConfig is immutable."""
        config = StereotypeConfig()
        with pytest.raises(AttributeError):
            config.sensitivity = 0.8  # type: ignore[misc]


class TestValidateBiasDetectionConfig:
    """Tests for validate_bias_detection_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = BiasDetectionConfig(
            protected_attributes=["gender"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        validate_bias_detection_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_bias_detection_config(None)  # type: ignore[arg-type]

    def test_empty_attributes_raises_error(self) -> None:
        """Test that empty protected_attributes raises ValueError."""
        config = BiasDetectionConfig(
            protected_attributes=[],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        with pytest.raises(ValueError, match="protected_attributes cannot be empty"):
            validate_bias_detection_config(config)

    def test_empty_metrics_raises_error(self) -> None:
        """Test that empty metrics raises ValueError."""
        config = BiasDetectionConfig(
            protected_attributes=["gender"],
            metrics=[],
        )
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            validate_bias_detection_config(config)

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        config = BiasDetectionConfig(
            protected_attributes=["gender"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
            threshold=0.0,
        )
        with pytest.raises(ValueError, match="threshold must be in"):
            validate_bias_detection_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold > 1 raises ValueError."""
        config = BiasDetectionConfig(
            protected_attributes=["gender"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
            threshold=1.5,
        )
        with pytest.raises(ValueError, match="threshold must be in"):
            validate_bias_detection_config(config)


class TestCreateBiasDetectionConfig:
    """Tests for create_bias_detection_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_bias_detection_config(
            protected_attributes=["gender"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        assert isinstance(config, BiasDetectionConfig)
        assert config.protected_attributes == ["gender"]

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_bias_detection_config(
            protected_attributes=["gender", "race"],
            metrics=[FairnessMetric.EQUALIZED_ODDS],
            threshold=0.05,
            intersectional=True,
        )
        assert config.threshold == pytest.approx(0.05)
        assert config.intersectional is True

    def test_empty_attributes_raises_error(self) -> None:
        """Test that empty attributes raises ValueError."""
        with pytest.raises(ValueError, match="protected_attributes cannot be empty"):
            create_bias_detection_config(
                protected_attributes=[],
                metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
            )


class TestValidateFairnessConstraint:
    """Tests for validate_fairness_constraint function."""

    def test_valid_constraint(self) -> None:
        """Test validation of valid constraint."""
        constraint = FairnessConstraint(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            group_comparison="pairwise",
        )
        validate_fairness_constraint(constraint)  # Should not raise

    def test_none_constraint_raises_error(self) -> None:
        """Test that None constraint raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_fairness_constraint(None)  # type: ignore[arg-type]

    def test_zero_threshold_raises_error(self) -> None:
        """Test that zero threshold raises ValueError."""
        constraint = FairnessConstraint(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            threshold=0.0,
        )
        with pytest.raises(ValueError, match="threshold must be positive"):
            validate_fairness_constraint(constraint)

    def test_invalid_comparison_raises_error(self) -> None:
        """Test that invalid group_comparison raises ValueError."""
        constraint = FairnessConstraint(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            group_comparison="invalid",
        )
        with pytest.raises(ValueError, match="group_comparison must be one of"):
            validate_fairness_constraint(constraint)

    def test_negative_slack_raises_error(self) -> None:
        """Test that negative slack raises ValueError."""
        constraint = FairnessConstraint(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            slack=-0.1,
        )
        with pytest.raises(ValueError, match="slack cannot be negative"):
            validate_fairness_constraint(constraint)

    def test_reference_comparison_valid(self) -> None:
        """Test that reference comparison is valid."""
        constraint = FairnessConstraint(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            group_comparison="reference",
        )
        validate_fairness_constraint(constraint)  # Should not raise


class TestCreateFairnessConstraint:
    """Tests for create_fairness_constraint function."""

    def test_creates_constraint(self) -> None:
        """Test that function creates a constraint."""
        constraint = create_fairness_constraint(FairnessMetric.EQUALIZED_ODDS)
        assert isinstance(constraint, FairnessConstraint)
        assert constraint.metric == FairnessMetric.EQUALIZED_ODDS

    def test_custom_values(self) -> None:
        """Test creating constraint with custom values."""
        constraint = create_fairness_constraint(
            FairnessMetric.DEMOGRAPHIC_PARITY,
            threshold=0.1,
            group_comparison="reference",
            slack=0.02,
        )
        assert constraint.threshold == pytest.approx(0.1)
        assert constraint.group_comparison == "reference"
        assert constraint.slack == pytest.approx(0.02)

    def test_invalid_comparison_raises_error(self) -> None:
        """Test that invalid comparison raises ValueError."""
        with pytest.raises(ValueError, match="group_comparison must be one of"):
            create_fairness_constraint(
                FairnessMetric.DEMOGRAPHIC_PARITY,
                group_comparison="invalid",
            )


class TestValidateStereotypeConfig:
    """Tests for validate_stereotype_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = StereotypeConfig(categories=["gender"], sensitivity=0.5)
        validate_stereotype_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_stereotype_config(None)  # type: ignore[arg-type]

    def test_invalid_sensitivity_raises_error(self) -> None:
        """Test that invalid sensitivity raises ValueError."""
        config = StereotypeConfig(sensitivity=1.5)
        with pytest.raises(ValueError, match="sensitivity must be in"):
            validate_stereotype_config(config)

    def test_negative_sensitivity_raises_error(self) -> None:
        """Test that negative sensitivity raises ValueError."""
        config = StereotypeConfig(sensitivity=-0.1)
        with pytest.raises(ValueError, match="sensitivity must be in"):
            validate_stereotype_config(config)

    def test_zero_context_window_raises_error(self) -> None:
        """Test that zero context_window raises ValueError."""
        config = StereotypeConfig(context_window=0)
        with pytest.raises(ValueError, match="context_window must be positive"):
            validate_stereotype_config(config)


class TestCreateStereotypeConfig:
    """Tests for create_stereotype_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_stereotype_config(categories=["gender"])
        assert isinstance(config, StereotypeConfig)
        assert config.categories == ["gender"]

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_stereotype_config()
        assert config.lexicon_path is None
        assert config.categories == []
        assert config.sensitivity == pytest.approx(0.5)
        assert config.context_window == 5

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = create_stereotype_config(
            lexicon_path="/path/to/lexicon",
            categories=["gender", "race"],
            sensitivity=0.8,
            context_window=10,
        )
        assert config.lexicon_path == "/path/to/lexicon"
        assert config.sensitivity == pytest.approx(0.8)

    def test_invalid_sensitivity_raises_error(self) -> None:
        """Test that invalid sensitivity raises ValueError."""
        with pytest.raises(ValueError, match="sensitivity must be in"):
            create_stereotype_config(sensitivity=2.0)


class TestListFairnessMetrics:
    """Tests for list_fairness_metrics function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        metrics = list_fairness_metrics()
        assert isinstance(metrics, list)

    def test_contains_expected_metrics(self) -> None:
        """Test that list contains expected metrics."""
        metrics = list_fairness_metrics()
        assert "demographic_parity" in metrics
        assert "equalized_odds" in metrics
        assert "calibration" in metrics
        assert "predictive_parity" in metrics

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        metrics = list_fairness_metrics()
        assert metrics == sorted(metrics)


class TestValidateFairnessMetric:
    """Tests for validate_fairness_metric function."""

    def test_valid_demographic_parity(self) -> None:
        """Test validation of demographic_parity."""
        assert validate_fairness_metric("demographic_parity") is True

    def test_valid_equalized_odds(self) -> None:
        """Test validation of equalized_odds."""
        assert validate_fairness_metric("equalized_odds") is True

    def test_invalid_metric(self) -> None:
        """Test validation of invalid metric."""
        assert validate_fairness_metric("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_fairness_metric("") is False


class TestGetFairnessMetric:
    """Tests for get_fairness_metric function."""

    def test_get_demographic_parity(self) -> None:
        """Test getting DEMOGRAPHIC_PARITY."""
        result = get_fairness_metric("demographic_parity")
        assert result == FairnessMetric.DEMOGRAPHIC_PARITY

    def test_get_equalized_odds(self) -> None:
        """Test getting EQUALIZED_ODDS."""
        result = get_fairness_metric("equalized_odds")
        assert result == FairnessMetric.EQUALIZED_ODDS

    def test_invalid_metric_raises_error(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="invalid fairness metric"):
            get_fairness_metric("invalid")


class TestListBiasTypes:
    """Tests for list_bias_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_bias_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_bias_types()
        assert "gender" in types
        assert "race" in types
        assert "age" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_bias_types()
        assert types == sorted(types)


class TestValidateBiasType:
    """Tests for validate_bias_type function."""

    def test_valid_gender(self) -> None:
        """Test validation of gender."""
        assert validate_bias_type("gender") is True

    def test_valid_race(self) -> None:
        """Test validation of race."""
        assert validate_bias_type("race") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_bias_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_bias_type("") is False


class TestGetBiasType:
    """Tests for get_bias_type function."""

    def test_get_gender(self) -> None:
        """Test getting GENDER."""
        result = get_bias_type("gender")
        assert result == BiasType.GENDER

    def test_get_age(self) -> None:
        """Test getting AGE."""
        result = get_bias_type("age")
        assert result == BiasType.AGE

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid bias type"):
            get_bias_type("invalid")


class TestListMitigationStrategies:
    """Tests for list_mitigation_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_mitigation_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_mitigation_strategies()
        assert "reweighting" in strategies
        assert "adversarial" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_mitigation_strategies()
        assert strategies == sorted(strategies)


class TestValidateMitigationStrategy:
    """Tests for validate_mitigation_strategy function."""

    def test_valid_reweighting(self) -> None:
        """Test validation of reweighting."""
        assert validate_mitigation_strategy("reweighting") is True

    def test_valid_adversarial(self) -> None:
        """Test validation of adversarial."""
        assert validate_mitigation_strategy("adversarial") is True

    def test_invalid_strategy(self) -> None:
        """Test validation of invalid strategy."""
        assert validate_mitigation_strategy("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_mitigation_strategy("") is False


class TestGetMitigationStrategy:
    """Tests for get_mitigation_strategy function."""

    def test_get_reweighting(self) -> None:
        """Test getting REWEIGHTING."""
        result = get_mitigation_strategy("reweighting")
        assert result == MitigationStrategy.REWEIGHTING

    def test_get_adversarial(self) -> None:
        """Test getting ADVERSARIAL."""
        result = get_mitigation_strategy("adversarial")
        assert result == MitigationStrategy.ADVERSARIAL

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid mitigation strategy"):
            get_mitigation_strategy("invalid")


class TestCalculateDemographicParity:
    """Tests for calculate_demographic_parity function."""

    def test_basic_calculation(self) -> None:
        """Test basic demographic parity calculation."""
        predictions = [1, 0, 1, 1, 0, 1]
        groups = ["A", "A", "A", "B", "B", "B"]
        result = calculate_demographic_parity(predictions, groups)
        assert "A" in result
        assert "B" in result
        assert result["A"] == pytest.approx(2 / 3)
        assert result["B"] == pytest.approx(2 / 3)

    def test_unequal_rates(self) -> None:
        """Test with unequal positive rates."""
        predictions = [1, 1, 1, 0, 0, 0]
        groups = ["A", "A", "A", "B", "B", "B"]
        result = calculate_demographic_parity(predictions, groups)
        assert result["A"] == pytest.approx(1.0)
        assert result["B"] == pytest.approx(0.0)

    def test_all_positive(self) -> None:
        """Test with all positive predictions."""
        predictions = [1, 1, 1, 1]
        groups = ["A", "A", "B", "B"]
        result = calculate_demographic_parity(predictions, groups)
        assert result["A"] == pytest.approx(1.0)
        assert result["B"] == pytest.approx(1.0)

    def test_all_negative(self) -> None:
        """Test with all negative predictions."""
        predictions = [0, 0, 0, 0]
        groups = ["A", "A", "B", "B"]
        result = calculate_demographic_parity(predictions, groups)
        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.0)

    def test_none_predictions_raises_error(self) -> None:
        """Test that None predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be None"):
            calculate_demographic_parity(None, ["A"])  # type: ignore[arg-type]

    def test_empty_predictions_raises_error(self) -> None:
        """Test that empty predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            calculate_demographic_parity([], ["A"])

    def test_none_groups_raises_error(self) -> None:
        """Test that None groups raises ValueError."""
        with pytest.raises(ValueError, match="groups cannot be None"):
            calculate_demographic_parity([1], None)  # type: ignore[arg-type]

    def test_empty_groups_raises_error(self) -> None:
        """Test that empty groups raises ValueError."""
        with pytest.raises(ValueError, match="groups cannot be empty"):
            calculate_demographic_parity([1], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_demographic_parity([1, 0], ["A"])


class TestCalculateEqualizedOdds:
    """Tests for calculate_equalized_odds function."""

    def test_basic_calculation(self) -> None:
        """Test basic equalized odds calculation."""
        predictions = [1, 0, 1, 1, 0, 0]
        labels = [1, 0, 0, 1, 0, 1]
        groups = ["A", "A", "A", "B", "B", "B"]
        result = calculate_equalized_odds(predictions, labels, groups)
        assert "A" in result
        assert "B" in result
        assert "tpr" in result["A"]
        assert "fpr" in result["A"]

    def test_perfect_classifier(self) -> None:
        """Test with perfect classifier."""
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        groups = ["A", "A", "B", "B"]
        result = calculate_equalized_odds(predictions, labels, groups)
        assert result["A"]["tpr"] == pytest.approx(1.0)
        assert result["A"]["fpr"] == pytest.approx(0.0)

    def test_none_predictions_raises_error(self) -> None:
        """Test that None predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be None"):
            calculate_equalized_odds(None, [1], ["A"])  # type: ignore[arg-type]

    def test_empty_predictions_raises_error(self) -> None:
        """Test that empty predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            calculate_equalized_odds([], [1], ["A"])

    def test_none_labels_raises_error(self) -> None:
        """Test that None labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be None"):
            calculate_equalized_odds([1], None, ["A"])  # type: ignore[arg-type]

    def test_empty_labels_raises_error(self) -> None:
        """Test that empty labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be empty"):
            calculate_equalized_odds([1], [], ["A"])

    def test_none_groups_raises_error(self) -> None:
        """Test that None groups raises ValueError."""
        with pytest.raises(ValueError, match="groups cannot be None"):
            calculate_equalized_odds([1], [1], None)  # type: ignore[arg-type]

    def test_empty_groups_raises_error(self) -> None:
        """Test that empty groups raises ValueError."""
        with pytest.raises(ValueError, match="groups cannot be empty"):
            calculate_equalized_odds([1], [1], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_equalized_odds([1, 0], [1], ["A"])

    def test_handles_no_positives(self) -> None:
        """Test handling case with no positive labels in group."""
        predictions = [1, 0]
        labels = [0, 0]
        groups = ["A", "A"]
        result = calculate_equalized_odds(predictions, labels, groups)
        assert result["A"]["tpr"] == pytest.approx(0.0)

    def test_handles_no_negatives(self) -> None:
        """Test handling case with no negative labels in group."""
        predictions = [1, 0]
        labels = [1, 1]
        groups = ["A", "A"]
        result = calculate_equalized_odds(predictions, labels, groups)
        assert result["A"]["fpr"] == pytest.approx(0.0)


class TestDetectStereotypes:
    """Tests for detect_stereotypes function."""

    def test_no_stereotypes(self) -> None:
        """Test with neutral text."""
        config = StereotypeConfig(categories=["gender"], sensitivity=0.5)
        result = detect_stereotypes("This is neutral text.", config)
        assert result == []

    def test_detects_gender_stereotype(self) -> None:
        """Test detection of gender stereotype."""
        config = StereotypeConfig(categories=["gender"], sensitivity=0.5)
        result = detect_stereotypes("Women are emotional by nature.", config)
        assert len(result) > 0
        assert result[0]["category"] == "gender"

    def test_detects_age_stereotype(self) -> None:
        """Test detection of age stereotype."""
        config = StereotypeConfig(categories=["age"], sensitivity=0.5)
        result = detect_stereotypes("Old people are forgetful.", config)
        assert len(result) > 0
        assert result[0]["category"] == "age"

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        config = StereotypeConfig(categories=["gender"])
        with pytest.raises(ValueError, match="text cannot be None"):
            detect_stereotypes(None, config)  # type: ignore[arg-type]

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            detect_stereotypes("text", None)  # type: ignore[arg-type]

    def test_empty_categories(self) -> None:
        """Test with empty categories."""
        config = StereotypeConfig(categories=[])
        result = detect_stereotypes("Women are emotional.", config)
        assert result == []

    def test_case_insensitive(self) -> None:
        """Test that detection is case insensitive."""
        config = StereotypeConfig(categories=["gender"], sensitivity=0.5)
        result = detect_stereotypes("WOMEN ARE EMOTIONAL by nature.", config)
        assert len(result) > 0


class TestCalculateDisparityScore:
    """Tests for calculate_disparity_score function."""

    def test_basic_calculation(self) -> None:
        """Test basic disparity score calculation."""
        group_rates = {"A": 0.8, "B": 0.7, "C": 0.6}
        result = calculate_disparity_score(group_rates)
        assert result == pytest.approx(0.2)

    def test_no_disparity(self) -> None:
        """Test with no disparity."""
        group_rates = {"A": 0.5, "B": 0.5, "C": 0.5}
        result = calculate_disparity_score(group_rates)
        assert result == pytest.approx(0.0)

    def test_two_groups(self) -> None:
        """Test with exactly two groups."""
        group_rates = {"A": 0.9, "B": 0.6}
        result = calculate_disparity_score(group_rates)
        assert result == pytest.approx(0.3)

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_disparity_score(None)  # type: ignore[arg-type]

    def test_single_group_raises_error(self) -> None:
        """Test that single group raises ValueError."""
        with pytest.raises(ValueError, match="must have at least 2 groups"):
            calculate_disparity_score({"A": 0.5})

    def test_empty_raises_error(self) -> None:
        """Test that empty dict raises ValueError."""
        with pytest.raises(ValueError, match="must have at least 2 groups"):
            calculate_disparity_score({})


class TestFormatBiasAudit:
    """Tests for format_bias_audit function."""

    def test_basic_formatting(self) -> None:
        """Test basic formatting."""
        result = BiasAuditResult(
            disparities={"gender": 0.15},
            scores_by_group={"male": 0.8, "female": 0.65},
            overall_fairness=0.75,
            recommendations=["Consider reweighting"],
        )
        formatted = format_bias_audit(result)
        assert "Bias Audit Report" in formatted
        assert "75.00%" in formatted
        assert "gender" in formatted

    def test_contains_recommendations(self) -> None:
        """Test that recommendations are included."""
        result = BiasAuditResult(
            disparities={"gender": 0.15},
            scores_by_group={},
            overall_fairness=0.75,
            recommendations=["Recommendation 1", "Recommendation 2"],
        )
        formatted = format_bias_audit(result)
        assert "Recommendation 1" in formatted
        assert "Recommendation 2" in formatted

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_bias_audit(None)  # type: ignore[arg-type]

    def test_empty_recommendations(self) -> None:
        """Test formatting with empty recommendations."""
        result = BiasAuditResult(
            disparities={},
            scores_by_group={},
            overall_fairness=0.9,
            recommendations=[],
        )
        formatted = format_bias_audit(result)
        assert "Recommendations" not in formatted


class TestGetRecommendedBiasConfig:
    """Tests for get_recommended_bias_config function."""

    def test_classification_task(self) -> None:
        """Test configuration for classification task."""
        config = get_recommended_bias_config("classification")
        assert FairnessMetric.DEMOGRAPHIC_PARITY in config.metrics
        assert "gender" in config.protected_attributes

    def test_generation_task(self) -> None:
        """Test configuration for generation task."""
        config = get_recommended_bias_config("generation")
        assert config.intersectional is True
        assert "nationality" in config.protected_attributes

    def test_hiring_task(self) -> None:
        """Test configuration for hiring task."""
        config = get_recommended_bias_config("hiring")
        assert config.threshold == pytest.approx(0.05)
        assert FairnessMetric.PREDICTIVE_PARITY in config.metrics

    def test_credit_task(self) -> None:
        """Test configuration for credit task."""
        config = get_recommended_bias_config("credit")
        assert FairnessMetric.CALIBRATION in config.metrics
        assert config.intersectional is True

    def test_healthcare_task(self) -> None:
        """Test configuration for healthcare task."""
        config = get_recommended_bias_config("healthcare")
        assert config.threshold == pytest.approx(0.03)
        assert FairnessMetric.EQUALIZED_ODDS in config.metrics

    def test_unknown_task_returns_default(self) -> None:
        """Test that unknown task returns default config."""
        config = get_recommended_bias_config("unknown_task")
        assert config is not None
        assert len(config.metrics) > 0

    def test_case_insensitive(self) -> None:
        """Test that task_type is case insensitive."""
        config1 = get_recommended_bias_config("classification")
        config2 = get_recommended_bias_config("CLASSIFICATION")
        assert config1.metrics == config2.metrics

    def test_none_task_raises_error(self) -> None:
        """Test that None task_type raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            get_recommended_bias_config(None)  # type: ignore[arg-type]

    def test_empty_task_raises_error(self) -> None:
        """Test that empty task_type raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_recommended_bias_config("")


class TestPropertyBased:
    """Property-based tests for bias functions."""

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=20)
    def test_disparity_score_non_negative(self, rates: list[float]) -> None:
        """Test that disparity score is non-negative."""
        group_rates = {f"group_{i}": r for i, r in enumerate(rates)}
        score = calculate_disparity_score(group_rates)
        assert score >= 0

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=20)
    def test_disparity_score_bounded(self, rates: list[float]) -> None:
        """Test that disparity score is at most 1."""
        group_rates = {f"group_{i}": r for i, r in enumerate(rates)}
        score = calculate_disparity_score(group_rates)
        assert score <= 1.0

    @given(
        st.lists(st.integers(min_value=0, max_value=1), min_size=2, max_size=20),
        st.lists(st.sampled_from(["A", "B", "C"]), min_size=2, max_size=20),
    )
    @settings(max_examples=20)
    def test_demographic_parity_rates_bounded(
        self, predictions: list[int], groups: list[str]
    ) -> None:
        """Test that demographic parity rates are in [0, 1]."""
        # Ensure same length
        min_len = min(len(predictions), len(groups))
        predictions = predictions[:min_len]
        groups = groups[:min_len]

        if min_len == 0:
            return

        result = calculate_demographic_parity(predictions, groups)
        for rate in result.values():
            assert 0 <= rate <= 1
