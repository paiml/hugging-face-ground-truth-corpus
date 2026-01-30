"""Tests for model calibration and uncertainty estimation functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.calibration import (
    VALID_CALIBRATION_METHODS,
    VALID_CALIBRATION_METRICS,
    VALID_UNCERTAINTY_TYPES,
    CalibrationConfig,
    CalibrationMethod,
    CalibrationMetric,
    CalibrationStats,
    ReliabilityDiagram,
    TemperatureConfig,
    UncertaintyResult,
    UncertaintyType,
    calculate_brier_score,
    calculate_ece,
    compute_reliability_diagram,
    create_calibration_config,
    create_temperature_config,
    create_uncertainty_result,
    estimate_uncertainty,
    format_calibration_stats,
    get_calibration_method,
    get_calibration_metric,
    get_recommended_calibration_config,
    get_uncertainty_type,
    list_calibration_methods,
    list_calibration_metrics,
    list_uncertainty_types,
    optimize_temperature,
    validate_calibration_config,
    validate_calibration_method,
    validate_calibration_metric,
    validate_calibration_stats,
    validate_temperature_config,
    validate_uncertainty_result,
    validate_uncertainty_type,
)


class TestCalibrationMethodEnum:
    """Tests for CalibrationMethod enum."""

    def test_temperature_value(self) -> None:
        """Test TEMPERATURE value."""
        assert CalibrationMethod.TEMPERATURE.value == "temperature"

    def test_platt_value(self) -> None:
        """Test PLATT value."""
        assert CalibrationMethod.PLATT.value == "platt"

    def test_isotonic_value(self) -> None:
        """Test ISOTONIC value."""
        assert CalibrationMethod.ISOTONIC.value == "isotonic"

    def test_histogram_value(self) -> None:
        """Test HISTOGRAM value."""
        assert CalibrationMethod.HISTOGRAM.value == "histogram"

    def test_focal_value(self) -> None:
        """Test FOCAL value."""
        assert CalibrationMethod.FOCAL.value == "focal"

    def test_valid_calibration_methods_frozenset(self) -> None:
        """Test that VALID_CALIBRATION_METHODS is a frozenset."""
        assert isinstance(VALID_CALIBRATION_METHODS, frozenset)
        assert len(VALID_CALIBRATION_METHODS) == 5


class TestUncertaintyTypeEnum:
    """Tests for UncertaintyType enum."""

    def test_aleatoric_value(self) -> None:
        """Test ALEATORIC value."""
        assert UncertaintyType.ALEATORIC.value == "aleatoric"

    def test_epistemic_value(self) -> None:
        """Test EPISTEMIC value."""
        assert UncertaintyType.EPISTEMIC.value == "epistemic"

    def test_predictive_value(self) -> None:
        """Test PREDICTIVE value."""
        assert UncertaintyType.PREDICTIVE.value == "predictive"

    def test_valid_uncertainty_types_frozenset(self) -> None:
        """Test that VALID_UNCERTAINTY_TYPES is a frozenset."""
        assert isinstance(VALID_UNCERTAINTY_TYPES, frozenset)
        assert len(VALID_UNCERTAINTY_TYPES) == 3


class TestCalibrationMetricEnum:
    """Tests for CalibrationMetric enum."""

    def test_ece_value(self) -> None:
        """Test ECE value."""
        assert CalibrationMetric.ECE.value == "ece"

    def test_mce_value(self) -> None:
        """Test MCE value."""
        assert CalibrationMetric.MCE.value == "mce"

    def test_brier_value(self) -> None:
        """Test BRIER value."""
        assert CalibrationMetric.BRIER.value == "brier"

    def test_reliability_value(self) -> None:
        """Test RELIABILITY value."""
        assert CalibrationMetric.RELIABILITY.value == "reliability"

    def test_valid_calibration_metrics_frozenset(self) -> None:
        """Test that VALID_CALIBRATION_METRICS is a frozenset."""
        assert isinstance(VALID_CALIBRATION_METRICS, frozenset)
        assert len(VALID_CALIBRATION_METRICS) == 4


class TestTemperatureConfig:
    """Tests for TemperatureConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TemperatureConfig()
        assert config.initial_temp == 1.0
        assert config.optimize is True
        assert config.lr == 0.01

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TemperatureConfig(
            initial_temp=2.0,
            optimize=False,
            lr=0.001,
        )
        assert config.initial_temp == 2.0
        assert config.optimize is False
        assert config.lr == 0.001

    def test_frozen(self) -> None:
        """Test that TemperatureConfig is immutable."""
        config = TemperatureConfig()
        with pytest.raises(AttributeError):
            config.initial_temp = 2.0  # type: ignore[misc]


class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass."""

    def test_creation_with_method(self) -> None:
        """Test creating CalibrationConfig with method."""
        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE)
        assert config.method == CalibrationMethod.TEMPERATURE

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE)
        assert config.temperature_config is None
        assert config.n_bins == 15
        assert config.validate_before is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        temp_config = TemperatureConfig(initial_temp=1.5)
        config = CalibrationConfig(
            method=CalibrationMethod.HISTOGRAM,
            temperature_config=temp_config,
            n_bins=20,
            validate_before=False,
        )
        assert config.method == CalibrationMethod.HISTOGRAM
        assert config.temperature_config is not None
        assert config.temperature_config.initial_temp == 1.5
        assert config.n_bins == 20
        assert config.validate_before is False

    def test_frozen(self) -> None:
        """Test that CalibrationConfig is immutable."""
        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE)
        with pytest.raises(AttributeError):
            config.n_bins = 20  # type: ignore[misc]


class TestUncertaintyResult:
    """Tests for UncertaintyResult dataclass."""

    def test_creation(self) -> None:
        """Test creating UncertaintyResult instance."""
        result = UncertaintyResult(
            mean=0.7,
            variance=0.05,
            confidence_interval=(0.5, 0.9),
            uncertainty_type=UncertaintyType.PREDICTIVE,
        )
        assert result.mean == pytest.approx(0.7)
        assert result.variance == pytest.approx(0.05)
        assert result.confidence_interval == (0.5, 0.9)
        assert result.uncertainty_type == UncertaintyType.PREDICTIVE

    def test_frozen(self) -> None:
        """Test that UncertaintyResult is immutable."""
        result = UncertaintyResult(0.7, 0.05, (0.5, 0.9), UncertaintyType.PREDICTIVE)
        with pytest.raises(AttributeError):
            result.mean = 0.8  # type: ignore[misc]


class TestReliabilityDiagram:
    """Tests for ReliabilityDiagram dataclass."""

    def test_creation(self) -> None:
        """Test creating ReliabilityDiagram instance."""
        diagram = ReliabilityDiagram(
            bin_confidences=(0.1, 0.5, 0.9),
            bin_accuracies=(0.15, 0.45, 0.85),
            bin_counts=(100, 200, 100),
            n_bins=3,
        )
        assert diagram.n_bins == 3
        assert len(diagram.bin_confidences) == 3
        assert len(diagram.bin_accuracies) == 3
        assert len(diagram.bin_counts) == 3

    def test_frozen(self) -> None:
        """Test that ReliabilityDiagram is immutable."""
        diagram = ReliabilityDiagram((0.5,), (0.5,), (100,), 1)
        with pytest.raises(AttributeError):
            diagram.n_bins = 2  # type: ignore[misc]


class TestCalibrationStats:
    """Tests for CalibrationStats dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating CalibrationStats with minimal data."""
        stats = CalibrationStats(
            ece=0.05,
            mce=0.15,
            brier_score=0.12,
            reliability_diagram=None,
            optimal_temperature=None,
        )
        assert stats.ece == pytest.approx(0.05)
        assert stats.mce == pytest.approx(0.15)
        assert stats.brier_score == pytest.approx(0.12)
        assert stats.reliability_diagram is None
        assert stats.optimal_temperature is None

    def test_creation_full(self) -> None:
        """Test creating CalibrationStats with all data."""
        diagram = ReliabilityDiagram(
            bin_confidences=(0.5,),
            bin_accuracies=(0.5,),
            bin_counts=(100,),
            n_bins=1,
        )
        stats = CalibrationStats(
            ece=0.05,
            mce=0.15,
            brier_score=0.12,
            reliability_diagram=diagram,
            optimal_temperature=1.5,
        )
        assert stats.reliability_diagram is not None
        assert stats.optimal_temperature == pytest.approx(1.5)

    def test_frozen(self) -> None:
        """Test that CalibrationStats is immutable."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, None)
        with pytest.raises(AttributeError):
            stats.ece = 0.1  # type: ignore[misc]


class TestValidateTemperatureConfig:
    """Tests for validate_temperature_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = TemperatureConfig()
        validate_temperature_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_temperature_config(None)  # type: ignore[arg-type]

    def test_non_positive_initial_temp_raises_error(self) -> None:
        """Test that non-positive initial_temp raises ValueError."""
        config = TemperatureConfig(initial_temp=0.0)
        with pytest.raises(ValueError, match="initial_temp must be positive"):
            validate_temperature_config(config)

    def test_negative_initial_temp_raises_error(self) -> None:
        """Test that negative initial_temp raises ValueError."""
        config = TemperatureConfig(initial_temp=-1.0)
        with pytest.raises(ValueError, match="initial_temp must be positive"):
            validate_temperature_config(config)

    def test_non_positive_lr_raises_error(self) -> None:
        """Test that non-positive lr raises ValueError."""
        config = TemperatureConfig(lr=0.0)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_temperature_config(config)

    def test_negative_lr_raises_error(self) -> None:
        """Test that negative lr raises ValueError."""
        config = TemperatureConfig(lr=-0.01)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_temperature_config(config)


class TestValidateCalibrationConfig:
    """Tests for validate_calibration_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE)
        validate_calibration_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_calibration_config(None)  # type: ignore[arg-type]

    def test_non_positive_n_bins_raises_error(self) -> None:
        """Test that non-positive n_bins raises ValueError."""
        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE, n_bins=0)
        with pytest.raises(ValueError, match="n_bins must be positive"):
            validate_calibration_config(config)

    def test_negative_n_bins_raises_error(self) -> None:
        """Test that negative n_bins raises ValueError."""
        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE, n_bins=-5)
        with pytest.raises(ValueError, match="n_bins must be positive"):
            validate_calibration_config(config)

    def test_validates_temperature_config_if_present(self) -> None:
        """Test that temperature config is validated if present."""
        temp_config = TemperatureConfig(initial_temp=-1.0)
        config = CalibrationConfig(
            method=CalibrationMethod.TEMPERATURE,
            temperature_config=temp_config,
        )
        with pytest.raises(ValueError, match="initial_temp must be positive"):
            validate_calibration_config(config)


class TestValidateUncertaintyResult:
    """Tests for validate_uncertainty_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = UncertaintyResult(0.5, 0.1, (0.3, 0.7), UncertaintyType.PREDICTIVE)
        validate_uncertainty_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_uncertainty_result(None)  # type: ignore[arg-type]

    def test_negative_variance_raises_error(self) -> None:
        """Test that negative variance raises ValueError."""
        result = UncertaintyResult(0.5, -0.1, (0.3, 0.7), UncertaintyType.PREDICTIVE)
        with pytest.raises(ValueError, match="variance cannot be negative"):
            validate_uncertainty_result(result)

    def test_invalid_confidence_interval_raises_error(self) -> None:
        """Test that invalid confidence interval raises ValueError."""
        result = UncertaintyResult(
            0.5,
            0.1,
            (0.7, 0.3),
            UncertaintyType.PREDICTIVE,  # lower > upper
        )
        with pytest.raises(ValueError, match=r"lower bound.*cannot be greater"):
            validate_uncertainty_result(result)

    def test_equal_bounds_is_valid(self) -> None:
        """Test that equal bounds are valid."""
        result = UncertaintyResult(0.5, 0.0, (0.5, 0.5), UncertaintyType.ALEATORIC)
        validate_uncertainty_result(result)  # Should not raise


class TestValidateCalibrationStats:
    """Tests for validate_calibration_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, 1.5)
        validate_calibration_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_calibration_stats(None)  # type: ignore[arg-type]

    def test_negative_ece_raises_error(self) -> None:
        """Test that negative ECE raises ValueError."""
        stats = CalibrationStats(-0.05, 0.15, 0.12, None, 1.5)
        with pytest.raises(ValueError, match="ece cannot be negative"):
            validate_calibration_stats(stats)

    def test_negative_mce_raises_error(self) -> None:
        """Test that negative MCE raises ValueError."""
        stats = CalibrationStats(0.05, -0.15, 0.12, None, 1.5)
        with pytest.raises(ValueError, match="mce cannot be negative"):
            validate_calibration_stats(stats)

    def test_brier_score_out_of_range_raises_error(self) -> None:
        """Test that Brier score out of range raises ValueError."""
        stats = CalibrationStats(0.05, 0.15, 1.5, None, 1.5)
        with pytest.raises(ValueError, match="brier_score must be between 0 and 1"):
            validate_calibration_stats(stats)

    def test_negative_brier_score_raises_error(self) -> None:
        """Test that negative Brier score raises ValueError."""
        stats = CalibrationStats(0.05, 0.15, -0.1, None, 1.5)
        with pytest.raises(ValueError, match="brier_score must be between 0 and 1"):
            validate_calibration_stats(stats)

    def test_non_positive_optimal_temp_raises_error(self) -> None:
        """Test that non-positive optimal_temperature raises ValueError."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, 0.0)
        with pytest.raises(ValueError, match="optimal_temperature must be positive"):
            validate_calibration_stats(stats)

    def test_negative_optimal_temp_raises_error(self) -> None:
        """Test that negative optimal_temperature raises ValueError."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, -1.0)
        with pytest.raises(ValueError, match="optimal_temperature must be positive"):
            validate_calibration_stats(stats)

    def test_none_optimal_temp_is_valid(self) -> None:
        """Test that None optimal_temperature is valid."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, None)
        validate_calibration_stats(stats)  # Should not raise


class TestCreateTemperatureConfig:
    """Tests for create_temperature_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_temperature_config()
        assert isinstance(config, TemperatureConfig)
        assert config.initial_temp == 1.0

    def test_with_custom_options(self) -> None:
        """Test config creation with custom options."""
        config = create_temperature_config(
            initial_temp=2.0,
            optimize=False,
            lr=0.001,
        )
        assert config.initial_temp == 2.0
        assert config.optimize is False
        assert config.lr == 0.001

    def test_invalid_initial_temp_raises_error(self) -> None:
        """Test that invalid initial_temp raises ValueError."""
        with pytest.raises(ValueError, match="initial_temp must be positive"):
            create_temperature_config(initial_temp=-1.0)

    def test_invalid_lr_raises_error(self) -> None:
        """Test that invalid lr raises ValueError."""
        with pytest.raises(ValueError, match="lr must be positive"):
            create_temperature_config(lr=0.0)


class TestCreateCalibrationConfig:
    """Tests for create_calibration_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_calibration_config(CalibrationMethod.TEMPERATURE)
        assert isinstance(config, CalibrationConfig)
        assert config.method == CalibrationMethod.TEMPERATURE

    def test_with_custom_options(self) -> None:
        """Test config creation with custom options."""
        temp_config = create_temperature_config(initial_temp=1.5)
        config = create_calibration_config(
            CalibrationMethod.HISTOGRAM,
            temperature_config=temp_config,
            n_bins=20,
            validate_before=False,
        )
        assert config.method == CalibrationMethod.HISTOGRAM
        assert config.n_bins == 20
        assert config.validate_before is False

    def test_invalid_n_bins_raises_error(self) -> None:
        """Test that invalid n_bins raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be positive"):
            create_calibration_config(CalibrationMethod.TEMPERATURE, n_bins=0)


class TestCreateUncertaintyResult:
    """Tests for create_uncertainty_result function."""

    def test_creates_result(self) -> None:
        """Test that function creates a result."""
        result = create_uncertainty_result(
            0.7, 0.05, (0.5, 0.9), UncertaintyType.PREDICTIVE
        )
        assert isinstance(result, UncertaintyResult)
        assert result.mean == 0.7

    def test_with_zero_variance(self) -> None:
        """Test result creation with zero variance."""
        result = create_uncertainty_result(
            0.5, 0.0, (0.5, 0.5), UncertaintyType.ALEATORIC
        )
        assert result.variance == 0.0

    def test_invalid_variance_raises_error(self) -> None:
        """Test that invalid variance raises ValueError."""
        with pytest.raises(ValueError, match="variance cannot be negative"):
            create_uncertainty_result(0.5, -0.1, (0.3, 0.7), UncertaintyType.PREDICTIVE)

    def test_invalid_interval_raises_error(self) -> None:
        """Test that invalid confidence interval raises ValueError."""
        with pytest.raises(ValueError, match=r"lower bound.*cannot be greater"):
            create_uncertainty_result(0.5, 0.1, (0.7, 0.3), UncertaintyType.PREDICTIVE)


class TestListCalibrationMethods:
    """Tests for list_calibration_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_calibration_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_calibration_methods()
        assert "temperature" in methods
        assert "platt" in methods
        assert "isotonic" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_calibration_methods()
        assert methods == sorted(methods)


class TestGetCalibrationMethod:
    """Tests for get_calibration_method function."""

    def test_get_temperature(self) -> None:
        """Test getting TEMPERATURE."""
        result = get_calibration_method("temperature")
        assert result == CalibrationMethod.TEMPERATURE

    def test_get_platt(self) -> None:
        """Test getting PLATT."""
        result = get_calibration_method("platt")
        assert result == CalibrationMethod.PLATT

    def test_get_isotonic(self) -> None:
        """Test getting ISOTONIC."""
        result = get_calibration_method("isotonic")
        assert result == CalibrationMethod.ISOTONIC

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid calibration method"):
            get_calibration_method("invalid")


class TestValidateCalibrationMethod:
    """Tests for validate_calibration_method function."""

    def test_valid_temperature(self) -> None:
        """Test validation of temperature."""
        assert validate_calibration_method("temperature") is True

    def test_valid_platt(self) -> None:
        """Test validation of platt."""
        assert validate_calibration_method("platt") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_calibration_method("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_calibration_method("") is False


class TestListUncertaintyTypes:
    """Tests for list_uncertainty_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_uncertainty_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_uncertainty_types()
        assert "aleatoric" in types
        assert "epistemic" in types
        assert "predictive" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_uncertainty_types()
        assert types == sorted(types)


class TestGetUncertaintyType:
    """Tests for get_uncertainty_type function."""

    def test_get_aleatoric(self) -> None:
        """Test getting ALEATORIC."""
        result = get_uncertainty_type("aleatoric")
        assert result == UncertaintyType.ALEATORIC

    def test_get_epistemic(self) -> None:
        """Test getting EPISTEMIC."""
        result = get_uncertainty_type("epistemic")
        assert result == UncertaintyType.EPISTEMIC

    def test_get_predictive(self) -> None:
        """Test getting PREDICTIVE."""
        result = get_uncertainty_type("predictive")
        assert result == UncertaintyType.PREDICTIVE

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid uncertainty type"):
            get_uncertainty_type("invalid")


class TestValidateUncertaintyType:
    """Tests for validate_uncertainty_type function."""

    def test_valid_aleatoric(self) -> None:
        """Test validation of aleatoric."""
        assert validate_uncertainty_type("aleatoric") is True

    def test_valid_epistemic(self) -> None:
        """Test validation of epistemic."""
        assert validate_uncertainty_type("epistemic") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_uncertainty_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_uncertainty_type("") is False


class TestListCalibrationMetrics:
    """Tests for list_calibration_metrics function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        metrics = list_calibration_metrics()
        assert isinstance(metrics, list)

    def test_contains_expected_metrics(self) -> None:
        """Test that list contains expected metrics."""
        metrics = list_calibration_metrics()
        assert "ece" in metrics
        assert "mce" in metrics
        assert "brier" in metrics

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        metrics = list_calibration_metrics()
        assert metrics == sorted(metrics)


class TestGetCalibrationMetric:
    """Tests for get_calibration_metric function."""

    def test_get_ece(self) -> None:
        """Test getting ECE."""
        result = get_calibration_metric("ece")
        assert result == CalibrationMetric.ECE

    def test_get_mce(self) -> None:
        """Test getting MCE."""
        result = get_calibration_metric("mce")
        assert result == CalibrationMetric.MCE

    def test_get_brier(self) -> None:
        """Test getting BRIER."""
        result = get_calibration_metric("brier")
        assert result == CalibrationMetric.BRIER

    def test_invalid_metric_raises_error(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="invalid calibration metric"):
            get_calibration_metric("invalid")


class TestValidateCalibrationMetric:
    """Tests for validate_calibration_metric function."""

    def test_valid_ece(self) -> None:
        """Test validation of ece."""
        assert validate_calibration_metric("ece") is True

    def test_valid_brier(self) -> None:
        """Test validation of brier."""
        assert validate_calibration_metric("brier") is True

    def test_invalid_metric(self) -> None:
        """Test validation of invalid metric."""
        assert validate_calibration_metric("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_calibration_metric("") is False


class TestCalculateECE:
    """Tests for calculate_ece function."""

    def test_perfect_calibration(self) -> None:
        """Test ECE for perfectly calibrated model."""
        # 50% confidence, 50% accuracy
        confidences = [0.5, 0.5, 0.5, 0.5]
        accuracies = [1, 0, 1, 0]
        ece = calculate_ece(confidences, accuracies, n_bins=2)
        assert ece == pytest.approx(0.0)

    def test_well_calibrated(self) -> None:
        """Test ECE for well-calibrated model."""
        # 80% confidence, 80% accuracy
        confidences = [0.8, 0.8, 0.8, 0.8, 0.8]
        accuracies = [1, 1, 1, 1, 0]
        ece = calculate_ece(confidences, accuracies, n_bins=2)
        assert ece == pytest.approx(0.0)

    def test_overconfident_model(self) -> None:
        """Test ECE for overconfident model."""
        # 90% confidence, 50% accuracy
        confidences = [0.9, 0.9, 0.9, 0.9]
        accuracies = [1, 1, 0, 0]
        ece = calculate_ece(confidences, accuracies, n_bins=2)
        assert ece > 0.0

    def test_underconfident_model(self) -> None:
        """Test ECE for underconfident model."""
        # 30% confidence, 80% accuracy
        confidences = [0.3, 0.3, 0.3, 0.3, 0.3]
        accuracies = [1, 1, 1, 1, 0]
        ece = calculate_ece(confidences, accuracies, n_bins=2)
        assert ece > 0.0

    def test_none_confidences_raises_error(self) -> None:
        """Test that None confidences raises ValueError."""
        with pytest.raises(ValueError, match="confidences cannot be None"):
            calculate_ece(None, [1, 0])  # type: ignore[arg-type]

    def test_none_accuracies_raises_error(self) -> None:
        """Test that None accuracies raises ValueError."""
        with pytest.raises(ValueError, match="accuracies cannot be None"):
            calculate_ece([0.5], None)  # type: ignore[arg-type]

    def test_empty_confidences_raises_error(self) -> None:
        """Test that empty confidences raises ValueError."""
        with pytest.raises(ValueError, match="confidences cannot be empty"):
            calculate_ece([], [])

    def test_empty_accuracies_raises_error(self) -> None:
        """Test that empty accuracies raises ValueError."""
        with pytest.raises(ValueError, match="accuracies cannot be empty"):
            calculate_ece([0.5], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_ece([0.5, 0.6], [1])

    def test_non_positive_n_bins_raises_error(self) -> None:
        """Test that non-positive n_bins raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be positive"):
            calculate_ece([0.5], [1], n_bins=0)

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidences must be in"):
            calculate_ece([1.5], [1])

    def test_negative_confidence_raises_error(self) -> None:
        """Test that negative confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidences must be in"):
            calculate_ece([-0.1], [1])


class TestCalculateBrierScore:
    """Tests for calculate_brier_score function."""

    def test_perfect_predictions(self) -> None:
        """Test Brier score for perfect predictions."""
        probs = [1.0, 0.0, 1.0]
        labels = [1, 0, 1]
        brier = calculate_brier_score(probs, labels)
        assert brier == pytest.approx(0.0)

    def test_worst_predictions(self) -> None:
        """Test Brier score for worst predictions."""
        probs = [0.0, 1.0]
        labels = [1, 0]
        brier = calculate_brier_score(probs, labels)
        assert brier == pytest.approx(1.0)

    def test_typical_predictions(self) -> None:
        """Test Brier score for typical predictions."""
        probs = [0.7, 0.3, 0.8]
        labels = [1, 0, 1]
        brier = calculate_brier_score(probs, labels)
        # (0.3^2 + 0.3^2 + 0.2^2) / 3 = 0.073...
        assert 0.0 < brier < 0.2

    def test_none_probabilities_raises_error(self) -> None:
        """Test that None probabilities raises ValueError."""
        with pytest.raises(ValueError, match="probabilities cannot be None"):
            calculate_brier_score(None, [1])  # type: ignore[arg-type]

    def test_none_labels_raises_error(self) -> None:
        """Test that None labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be None"):
            calculate_brier_score([0.5], None)  # type: ignore[arg-type]

    def test_empty_probabilities_raises_error(self) -> None:
        """Test that empty probabilities raises ValueError."""
        with pytest.raises(ValueError, match="probabilities cannot be empty"):
            calculate_brier_score([], [])

    def test_empty_labels_raises_error(self) -> None:
        """Test that empty labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be empty"):
            calculate_brier_score([0.5], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_brier_score([0.5, 0.6], [1])

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="probabilities must be in"):
            calculate_brier_score([1.5], [1])

    def test_negative_probability_raises_error(self) -> None:
        """Test that negative probability raises ValueError."""
        with pytest.raises(ValueError, match="probabilities must be in"):
            calculate_brier_score([-0.1], [1])

    def test_invalid_label_raises_error(self) -> None:
        """Test that invalid label raises ValueError."""
        with pytest.raises(ValueError, match="labels must be 0 or 1"):
            calculate_brier_score([0.5], [2])


class TestComputeReliabilityDiagram:
    """Tests for compute_reliability_diagram function."""

    def test_basic_diagram(self) -> None:
        """Test basic reliability diagram computation."""
        confidences = [0.2, 0.3, 0.7, 0.8]
        accuracies = [0, 0, 1, 1]
        diagram = compute_reliability_diagram(confidences, accuracies, n_bins=2)
        assert diagram.n_bins == 2
        assert len(diagram.bin_confidences) == 2
        assert len(diagram.bin_accuracies) == 2
        assert len(diagram.bin_counts) == 2

    def test_empty_bins(self) -> None:
        """Test diagram with empty bins."""
        # All samples in high confidence bin
        confidences = [0.9, 0.95, 0.85]
        accuracies = [1, 1, 1]
        diagram = compute_reliability_diagram(confidences, accuracies, n_bins=3)
        assert diagram.n_bins == 3
        # Low confidence bins should have 0 count
        assert any(c == 0 for c in diagram.bin_counts)

    def test_none_confidences_raises_error(self) -> None:
        """Test that None confidences raises ValueError."""
        with pytest.raises(ValueError, match="confidences cannot be None"):
            compute_reliability_diagram(None, [1])  # type: ignore[arg-type]

    def test_none_accuracies_raises_error(self) -> None:
        """Test that None accuracies raises ValueError."""
        with pytest.raises(ValueError, match="accuracies cannot be None"):
            compute_reliability_diagram([0.5], None)  # type: ignore[arg-type]

    def test_empty_confidences_raises_error(self) -> None:
        """Test that empty confidences raises ValueError."""
        with pytest.raises(ValueError, match="confidences cannot be empty"):
            compute_reliability_diagram([], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            compute_reliability_diagram([0.5, 0.6], [1])

    def test_non_positive_n_bins_raises_error(self) -> None:
        """Test that non-positive n_bins raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be positive"):
            compute_reliability_diagram([0.5], [1], n_bins=0)


class TestOptimizeTemperature:
    """Tests for optimize_temperature function."""

    def test_basic_optimization(self) -> None:
        """Test basic temperature optimization."""
        logits = [[2.0, 0.5], [0.3, 2.5], [2.1, 0.2]]
        labels = [0, 1, 0]
        temp = optimize_temperature(logits, labels)
        assert 0.1 <= temp <= 10.0

    def test_with_config(self) -> None:
        """Test optimization with custom config."""
        logits = [[2.0, 0.5], [0.3, 2.5]]
        labels = [0, 1]
        config = TemperatureConfig(initial_temp=1.5)
        temp = optimize_temperature(logits, labels, config)
        assert temp > 0.0

    def test_no_optimization(self) -> None:
        """Test with optimization disabled."""
        logits = [[2.0, 0.5], [0.3, 2.5]]
        labels = [0, 1]
        config = TemperatureConfig(initial_temp=2.5, optimize=False)
        temp = optimize_temperature(logits, labels, config)
        assert temp == 2.5

    def test_none_logits_raises_error(self) -> None:
        """Test that None logits raises ValueError."""
        with pytest.raises(ValueError, match="logits cannot be None"):
            optimize_temperature(None, [0])  # type: ignore[arg-type]

    def test_none_labels_raises_error(self) -> None:
        """Test that None labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be None"):
            optimize_temperature([[1.0, 0.0]], None)  # type: ignore[arg-type]

    def test_empty_logits_raises_error(self) -> None:
        """Test that empty logits raises ValueError."""
        with pytest.raises(ValueError, match="logits cannot be empty"):
            optimize_temperature([], [])

    def test_empty_labels_raises_error(self) -> None:
        """Test that empty labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be empty"):
            optimize_temperature([[1.0, 0.0]], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            optimize_temperature([[1.0, 0.0], [0.5, 0.5]], [0])


class TestEstimateUncertainty:
    """Tests for estimate_uncertainty function."""

    def test_basic_estimation(self) -> None:
        """Test basic uncertainty estimation."""
        preds = [0.68, 0.72, 0.65, 0.70, 0.71]
        result = estimate_uncertainty(preds)
        assert isinstance(result, UncertaintyResult)
        assert 0.65 <= result.mean <= 0.72
        assert result.variance >= 0.0

    def test_single_prediction(self) -> None:
        """Test uncertainty with single prediction."""
        preds = [0.7]
        result = estimate_uncertainty(preds)
        assert result.mean == 0.7
        assert result.variance == 0.0
        assert result.confidence_interval == (0.7, 0.7)

    def test_custom_uncertainty_type(self) -> None:
        """Test with custom uncertainty type."""
        preds = [0.5, 0.6, 0.55]
        result = estimate_uncertainty(preds, uncertainty_type=UncertaintyType.EPISTEMIC)
        assert result.uncertainty_type == UncertaintyType.EPISTEMIC

    def test_custom_confidence_level(self) -> None:
        """Test with custom confidence level."""
        preds = [0.5, 0.6, 0.55]
        result_95 = estimate_uncertainty(preds, confidence_level=0.95)
        result_99 = estimate_uncertainty(preds, confidence_level=0.99)
        # 99% CI should be wider
        width_95 = result_95.confidence_interval[1] - result_95.confidence_interval[0]
        width_99 = result_99.confidence_interval[1] - result_99.confidence_interval[0]
        assert width_99 >= width_95

    def test_none_predictions_raises_error(self) -> None:
        """Test that None predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be None"):
            estimate_uncertainty(None)  # type: ignore[arg-type]

    def test_empty_predictions_raises_error(self) -> None:
        """Test that empty predictions raises ValueError."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            estimate_uncertainty([])

    def test_invalid_confidence_level_raises_error(self) -> None:
        """Test that invalid confidence level raises ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be in"):
            estimate_uncertainty([0.5], confidence_level=1.0)

    def test_zero_confidence_level_raises_error(self) -> None:
        """Test that zero confidence level raises ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be in"):
            estimate_uncertainty([0.5], confidence_level=0.0)


class TestFormatCalibrationStats:
    """Tests for format_calibration_stats function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, 1.5)
        formatted = format_calibration_stats(stats)
        assert "Calibration Statistics" in formatted
        assert "ECE" in formatted
        assert "0.05" in formatted

    def test_format_with_optimal_temperature(self) -> None:
        """Test formatting with optimal temperature."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, 1.5)
        formatted = format_calibration_stats(stats)
        assert "Optimal Temperature" in formatted
        assert "1.5" in formatted

    def test_format_without_optimal_temperature(self) -> None:
        """Test formatting without optimal temperature."""
        stats = CalibrationStats(0.05, 0.15, 0.12, None, None)
        formatted = format_calibration_stats(stats)
        assert "Optimal Temperature" not in formatted

    def test_format_with_reliability_diagram(self) -> None:
        """Test formatting with reliability diagram."""
        diagram = ReliabilityDiagram(
            bin_confidences=(0.3, 0.7),
            bin_accuracies=(0.35, 0.65),
            bin_counts=(50, 50),
            n_bins=2,
        )
        stats = CalibrationStats(0.05, 0.15, 0.12, diagram, None)
        formatted = format_calibration_stats(stats)
        assert "Reliability Diagram" in formatted
        assert "Bin" in formatted

    def test_excellent_calibration_interpretation(self) -> None:
        """Test interpretation for excellent calibration."""
        stats = CalibrationStats(0.03, 0.10, 0.08, None, None)
        formatted = format_calibration_stats(stats)
        assert "Excellent calibration" in formatted

    def test_good_calibration_interpretation(self) -> None:
        """Test interpretation for good calibration."""
        stats = CalibrationStats(0.08, 0.20, 0.15, None, None)
        formatted = format_calibration_stats(stats)
        assert "Good calibration" in formatted

    def test_moderate_calibration_interpretation(self) -> None:
        """Test interpretation for moderate calibration."""
        stats = CalibrationStats(0.12, 0.25, 0.18, None, None)
        formatted = format_calibration_stats(stats)
        assert "Moderate calibration" in formatted

    def test_poor_calibration_interpretation(self) -> None:
        """Test interpretation for poor calibration."""
        stats = CalibrationStats(0.20, 0.35, 0.25, None, None)
        formatted = format_calibration_stats(stats)
        assert "Poor calibration" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_calibration_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedCalibrationConfig:
    """Tests for get_recommended_calibration_config function."""

    def test_classification_config(self) -> None:
        """Test classification recommended config."""
        config = get_recommended_calibration_config("classification")
        assert "method" in config
        assert config["method"] == CalibrationMethod.TEMPERATURE

    def test_multi_class_config(self) -> None:
        """Test multi_class recommended config."""
        config = get_recommended_calibration_config("multi_class")
        assert config["n_bins"] == 20

    def test_regression_config(self) -> None:
        """Test regression recommended config."""
        config = get_recommended_calibration_config("regression")
        assert config["method"] == CalibrationMethod.ISOTONIC

    def test_llm_config(self) -> None:
        """Test LLM recommended config."""
        config = get_recommended_calibration_config("llm")
        assert config["method"] == CalibrationMethod.TEMPERATURE
        assert "uncertainty_type" in config

    def test_medical_config(self) -> None:
        """Test medical recommended config."""
        config = get_recommended_calibration_config("medical")
        assert config["method"] == CalibrationMethod.PLATT
        assert len(config["metrics"]) >= 3

    def test_unknown_model_type_returns_base(self) -> None:
        """Test that unknown model type returns base config."""
        config = get_recommended_calibration_config("unknown")
        assert "method" in config
        assert config["method"] == CalibrationMethod.TEMPERATURE

    def test_case_insensitive(self) -> None:
        """Test that model type is case insensitive."""
        config1 = get_recommended_calibration_config("CLASSIFICATION")
        config2 = get_recommended_calibration_config("classification")
        assert config1 == config2

    def test_with_small_dataset_size(self) -> None:
        """Test with small dataset size adjustment."""
        config = get_recommended_calibration_config("classification", dataset_size=500)
        assert config["n_bins"] <= 10

    def test_with_large_dataset_size(self) -> None:
        """Test with large dataset size adjustment."""
        config = get_recommended_calibration_config(
            "classification", dataset_size=50000
        )
        assert config["n_bins"] >= 20

    def test_none_model_type_raises_error(self) -> None:
        """Test that None model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be None"):
            get_recommended_calibration_config(None)  # type: ignore[arg-type]

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_calibration_config("")


class TestPropertyBased:
    """Property-based tests for calibration functions."""

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=10)
    def test_brier_score_bounded(self, probabilities: list[float]) -> None:
        """Test that Brier score is always between 0 and 1."""
        # Generate labels (all 1s for simplicity)
        labels = [1] * len(probabilities)
        brier = calculate_brier_score(probabilities, labels)
        assert 0.0 <= brier <= 1.0

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=10)
    def test_ece_bounded(self, confidences: list[float]) -> None:
        """Test that ECE is always non-negative."""
        accuracies = [1 if c >= 0.5 else 0 for c in confidences]
        ece = calculate_ece(confidences, accuracies, n_bins=5)
        assert ece >= 0.0

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_uncertainty_variance_nonnegative(self, predictions: list[float]) -> None:
        """Test that uncertainty variance is always non-negative."""
        result = estimate_uncertainty(predictions)
        assert result.variance >= 0.0

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_uncertainty_mean_in_range(self, predictions: list[float]) -> None:
        """Test that uncertainty mean is within data range."""
        result = estimate_uncertainty(predictions)
        assert min(predictions) <= result.mean <= max(predictions)

    @given(st.integers(min_value=1, max_value=30))
    @settings(max_examples=10)
    def test_reliability_diagram_bins_count(self, n_bins: int) -> None:
        """Test that reliability diagram has correct number of bins."""
        confidences = [0.1, 0.5, 0.9]
        accuracies = [0, 1, 1]
        diagram = compute_reliability_diagram(confidences, accuracies, n_bins=n_bins)
        assert diagram.n_bins == n_bins
        assert len(diagram.bin_confidences) == n_bins
        assert len(diagram.bin_accuracies) == n_bins
        assert len(diagram.bin_counts) == n_bins

    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=10)
    def test_temperature_config_valid_temp(self, temp: float) -> None:
        """Test that valid temperatures create valid configs."""
        config = create_temperature_config(initial_temp=temp)
        assert config.initial_temp == temp
        validate_temperature_config(config)  # Should not raise

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10)
    def test_calibration_config_valid_bins(self, n_bins: int) -> None:
        """Test that valid n_bins create valid configs."""
        config = create_calibration_config(
            CalibrationMethod.HISTOGRAM,
            n_bins=n_bins,
        )
        assert config.n_bins == n_bins
        validate_calibration_config(config)  # Should not raise
