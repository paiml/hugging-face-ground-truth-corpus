"""Tests for training callbacks functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.training.callbacks import (
    CallbackMetrics,
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    MetricMode,
    create_early_stopping_callback,
    create_logging_callback,
    get_recommended_callbacks,
    list_callback_types,
    should_stop_early,
    validate_checkpoint_config,
    validate_early_stopping_config,
    validate_logging_config,
)


class TestMetricMode:
    """Tests for MetricMode enum."""

    def test_min_mode(self) -> None:
        """Test MIN mode value."""
        assert MetricMode.MIN.value == "min"

    def test_max_mode(self) -> None:
        """Test MAX mode value."""
        assert MetricMode.MAX.value == "max"


class TestEarlyStoppingConfig:
    """Tests for EarlyStoppingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EarlyStoppingConfig()
        assert config.patience == 3
        assert config.threshold == 0.0
        assert config.metric == "eval_loss"
        assert config.mode == MetricMode.MIN

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = EarlyStoppingConfig(
            patience=5,
            threshold=0.01,
            metric="accuracy",
            mode=MetricMode.MAX,
        )
        assert config.patience == 5
        assert config.threshold == 0.01
        assert config.metric == "accuracy"
        assert config.mode == MetricMode.MAX

    def test_frozen(self) -> None:
        """Test that EarlyStoppingConfig is immutable."""
        config = EarlyStoppingConfig()
        with pytest.raises(AttributeError):
            config.patience = 10  # type: ignore[misc]


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.log_every_n_steps == 100
        assert config.log_predictions is False
        assert config.log_gradients is False
        assert config.max_samples_to_log == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LoggingConfig(
            log_every_n_steps=50,
            log_predictions=True,
            log_gradients=True,
            max_samples_to_log=5,
        )
        assert config.log_every_n_steps == 50
        assert config.log_predictions is True
        assert config.log_gradients is True
        assert config.max_samples_to_log == 5

    def test_frozen(self) -> None:
        """Test that LoggingConfig is immutable."""
        config = LoggingConfig()
        with pytest.raises(AttributeError):
            config.log_every_n_steps = 50  # type: ignore[misc]


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CheckpointConfig()
        assert config.save_every_n_steps == 500
        assert config.save_total_limit is None
        assert config.save_on_each_node is False
        assert config.metric_for_best_model == "eval_loss"
        assert config.greater_is_better is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CheckpointConfig(
            save_every_n_steps=1000,
            save_total_limit=5,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        assert config.save_every_n_steps == 1000
        assert config.save_total_limit == 5
        assert config.metric_for_best_model == "accuracy"
        assert config.greater_is_better is True

    def test_frozen(self) -> None:
        """Test that CheckpointConfig is immutable."""
        config = CheckpointConfig()
        with pytest.raises(AttributeError):
            config.save_every_n_steps = 1000  # type: ignore[misc]


class TestCallbackMetrics:
    """Tests for CallbackMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = CallbackMetrics()
        assert metrics.best_metric is None
        assert metrics.best_step == 0
        assert metrics.epochs_without_improvement == 0
        assert metrics.total_steps == 0

    def test_custom_values(self) -> None:
        """Test custom metric values."""
        metrics = CallbackMetrics(
            best_metric=0.5,
            best_step=100,
            epochs_without_improvement=2,
            total_steps=500,
        )
        assert metrics.best_metric == 0.5
        assert metrics.best_step == 100
        assert metrics.epochs_without_improvement == 2
        assert metrics.total_steps == 500


class TestValidateEarlyStoppingConfig:
    """Tests for validate_early_stopping_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = EarlyStoppingConfig(patience=3, threshold=0.01)
        validate_early_stopping_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_early_stopping_config(None)  # type: ignore[arg-type]

    def test_zero_patience_raises_error(self) -> None:
        """Test that zero patience raises ValueError."""
        config = EarlyStoppingConfig(patience=0)
        with pytest.raises(ValueError, match="patience must be positive"):
            validate_early_stopping_config(config)

    def test_negative_patience_raises_error(self) -> None:
        """Test that negative patience raises ValueError."""
        config = EarlyStoppingConfig(patience=-1)
        with pytest.raises(ValueError, match="patience must be positive"):
            validate_early_stopping_config(config)

    def test_negative_threshold_raises_error(self) -> None:
        """Test that negative threshold raises ValueError."""
        config = EarlyStoppingConfig(threshold=-0.01)
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            validate_early_stopping_config(config)

    def test_empty_metric_raises_error(self) -> None:
        """Test that empty metric raises ValueError."""
        config = EarlyStoppingConfig(metric="")
        with pytest.raises(ValueError, match="metric cannot be empty"):
            validate_early_stopping_config(config)

    def test_whitespace_metric_raises_error(self) -> None:
        """Test that whitespace-only metric raises ValueError."""
        config = EarlyStoppingConfig(metric="   ")
        with pytest.raises(ValueError, match="metric cannot be empty"):
            validate_early_stopping_config(config)


class TestValidateLoggingConfig:
    """Tests for validate_logging_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = LoggingConfig(log_every_n_steps=100)
        validate_logging_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_logging_config(None)  # type: ignore[arg-type]

    def test_zero_log_steps_raises_error(self) -> None:
        """Test that zero log_every_n_steps raises ValueError."""
        config = LoggingConfig(log_every_n_steps=0)
        with pytest.raises(ValueError, match="log_every_n_steps must be positive"):
            validate_logging_config(config)

    def test_zero_max_samples_raises_error(self) -> None:
        """Test that zero max_samples_to_log raises ValueError."""
        config = LoggingConfig(max_samples_to_log=0)
        with pytest.raises(ValueError, match="max_samples_to_log must be positive"):
            validate_logging_config(config)


class TestValidateCheckpointConfig:
    """Tests for validate_checkpoint_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = CheckpointConfig(save_every_n_steps=500)
        validate_checkpoint_config(config)  # Should not raise

    def test_valid_config_with_limit(self) -> None:
        """Test validation with save_total_limit set."""
        config = CheckpointConfig(save_every_n_steps=500, save_total_limit=3)
        validate_checkpoint_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_checkpoint_config(None)  # type: ignore[arg-type]

    def test_zero_save_steps_raises_error(self) -> None:
        """Test that zero save_every_n_steps raises ValueError."""
        config = CheckpointConfig(save_every_n_steps=0)
        with pytest.raises(ValueError, match="save_every_n_steps must be positive"):
            validate_checkpoint_config(config)

    def test_zero_total_limit_raises_error(self) -> None:
        """Test that zero save_total_limit raises ValueError."""
        config = CheckpointConfig(save_total_limit=0)
        with pytest.raises(ValueError, match="save_total_limit must be positive"):
            validate_checkpoint_config(config)

    def test_negative_total_limit_raises_error(self) -> None:
        """Test that negative save_total_limit raises ValueError."""
        config = CheckpointConfig(save_total_limit=-1)
        with pytest.raises(ValueError, match="save_total_limit must be positive"):
            validate_checkpoint_config(config)


class TestShouldStopEarly:
    """Tests for should_stop_early function."""

    def test_first_evaluation(self) -> None:
        """Test first evaluation initializes best metric."""
        config = EarlyStoppingConfig(patience=3, mode=MetricMode.MIN)
        should_stop, best, epochs = should_stop_early(0.5, None, 0, config)
        assert should_stop is False
        assert best == 0.5
        assert epochs == 0

    def test_metric_improved_min_mode(self) -> None:
        """Test when metric improves in MIN mode."""
        config = EarlyStoppingConfig(patience=3, threshold=0.0, mode=MetricMode.MIN)
        should_stop, best, epochs = should_stop_early(0.3, 0.5, 2, config)
        assert should_stop is False
        assert best == 0.3
        assert epochs == 0

    def test_metric_improved_max_mode(self) -> None:
        """Test when metric improves in MAX mode."""
        config = EarlyStoppingConfig(patience=3, threshold=0.0, mode=MetricMode.MAX)
        should_stop, best, epochs = should_stop_early(0.9, 0.7, 2, config)
        assert should_stop is False
        assert best == 0.9
        assert epochs == 0

    def test_metric_not_improved(self) -> None:
        """Test when metric does not improve."""
        config = EarlyStoppingConfig(patience=3, mode=MetricMode.MIN)
        should_stop, best, epochs = should_stop_early(0.6, 0.5, 0, config)
        assert should_stop is False
        assert best == 0.5
        assert epochs == 1

    def test_stop_after_patience_exceeded(self) -> None:
        """Test stopping when patience is exceeded."""
        config = EarlyStoppingConfig(patience=3, mode=MetricMode.MIN)
        should_stop, best, epochs = should_stop_early(0.6, 0.5, 2, config)
        assert should_stop is True
        assert best == 0.5
        assert epochs == 3

    def test_threshold_prevents_false_improvement(self) -> None:
        """Test that threshold prevents tiny improvements."""
        config = EarlyStoppingConfig(patience=3, threshold=0.1, mode=MetricMode.MIN)
        # Small improvement less than threshold
        should_stop, best, epochs = should_stop_early(0.45, 0.5, 0, config)
        assert should_stop is False
        assert best == 0.5  # Not updated
        assert epochs == 1

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            should_stop_early(0.5, None, 0, None)  # type: ignore[arg-type]

    @given(
        current=st.floats(min_value=0.0, max_value=1.0),
        best=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20)
    def test_epochs_increase_or_reset(self, current: float, best: float) -> None:
        """Test that epochs either increase or reset to 0."""
        config = EarlyStoppingConfig(patience=5, threshold=0.0, mode=MetricMode.MIN)
        _, _, epochs = should_stop_early(current, best, 2, config)
        # Epochs should either be 0 (improved) or 3 (not improved)
        assert epochs in {0, 3}


class TestCreateEarlyStoppingCallback:
    """Tests for create_early_stopping_callback function."""

    def test_creates_callback(self) -> None:
        """Test that callback is created successfully."""
        config = EarlyStoppingConfig(patience=3)
        callback = create_early_stopping_callback(config)
        assert callback is not None

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        config = EarlyStoppingConfig(patience=0)
        with pytest.raises(ValueError):
            create_early_stopping_callback(config)


class TestCreateLoggingCallback:
    """Tests for create_logging_callback function."""

    def test_creates_callback(self) -> None:
        """Test that callback is created successfully."""
        config = LoggingConfig(log_every_n_steps=100)
        callback = create_logging_callback(config)
        assert callback is not None

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        config = LoggingConfig(log_every_n_steps=0)
        with pytest.raises(ValueError):
            create_logging_callback(config)

    def test_callback_has_config(self) -> None:
        """Test that callback stores the config."""
        config = LoggingConfig(log_every_n_steps=50)
        callback = create_logging_callback(config)
        assert hasattr(callback, "config")
        assert callback.config.log_every_n_steps == 50


class TestGetRecommendedCallbacks:
    """Tests for get_recommended_callbacks function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        callbacks = get_recommended_callbacks()
        assert isinstance(callbacks, list)

    def test_includes_early_stopping_by_default(self) -> None:
        """Test that early stopping is included by default."""
        callbacks = get_recommended_callbacks()
        types = [cb["type"] for cb in callbacks]
        assert "early_stopping" in types

    def test_includes_logging_by_default(self) -> None:
        """Test that logging is included by default."""
        callbacks = get_recommended_callbacks()
        types = [cb["type"] for cb in callbacks]
        assert "logging" in types

    def test_disable_early_stopping(self) -> None:
        """Test disabling early stopping."""
        callbacks = get_recommended_callbacks(enable_early_stopping=False)
        types = [cb["type"] for cb in callbacks]
        assert "early_stopping" not in types

    def test_disable_logging(self) -> None:
        """Test disabling logging."""
        callbacks = get_recommended_callbacks(enable_logging=False)
        types = [cb["type"] for cb in callbacks]
        assert "logging" not in types

    def test_all_disabled(self) -> None:
        """Test with all callbacks disabled."""
        callbacks = get_recommended_callbacks(
            enable_early_stopping=False, enable_logging=False
        )
        assert callbacks == []

    def test_callback_config_format(self) -> None:
        """Test that callbacks have correct config format."""
        callbacks = get_recommended_callbacks()
        for cb in callbacks:
            assert "type" in cb
            assert "config" in cb
            assert isinstance(cb["config"], dict)


class TestListCallbackTypes:
    """Tests for list_callback_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_callback_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected callback types."""
        types = list_callback_types()
        assert "early_stopping" in types
        assert "logging" in types
        assert "checkpoint" in types

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_callback_types()
        assert types == sorted(types)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        types = list_callback_types()
        assert all(isinstance(t, str) for t in types)
