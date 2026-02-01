"""Tests for HuggingFace Hub telemetry functionality."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from hf_gtc.hub.telemetry import (
    VALID_EXPORT_FORMATS,
    VALID_LOG_LEVELS,
    VALID_METRIC_TYPES,
    ExportFormat,
    LogConfig,
    LogLevel,
    MetricConfig,
    MetricType,
    MetricValue,
    TelemetryConfig,
    TelemetryStats,
    aggregate_metrics,
    calculate_percentiles,
    create_log_config,
    create_metric_config,
    create_telemetry_config,
    export_telemetry,
    format_log_message,
    format_telemetry_stats,
    get_export_format,
    get_log_level,
    get_metric_type,
    get_recommended_telemetry_config,
    list_export_formats,
    list_log_levels,
    list_metric_types,
    record_metric,
    validate_log_config,
    validate_metric_config,
    validate_telemetry_config,
    validate_telemetry_stats,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_debug_value(self) -> None:
        """Test DEBUG level value."""
        assert LogLevel.DEBUG.value == "debug"

    def test_info_value(self) -> None:
        """Test INFO level value."""
        assert LogLevel.INFO.value == "info"

    def test_warning_value(self) -> None:
        """Test WARNING level value."""
        assert LogLevel.WARNING.value == "warning"

    def test_error_value(self) -> None:
        """Test ERROR level value."""
        assert LogLevel.ERROR.value == "error"

    def test_critical_value(self) -> None:
        """Test CRITICAL level value."""
        assert LogLevel.CRITICAL.value == "critical"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_counter_value(self) -> None:
        """Test COUNTER type value."""
        assert MetricType.COUNTER.value == "counter"

    def test_gauge_value(self) -> None:
        """Test GAUGE type value."""
        assert MetricType.GAUGE.value == "gauge"

    def test_histogram_value(self) -> None:
        """Test HISTOGRAM type value."""
        assert MetricType.HISTOGRAM.value == "histogram"

    def test_summary_value(self) -> None:
        """Test SUMMARY type value."""
        assert MetricType.SUMMARY.value == "summary"


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_json_value(self) -> None:
        """Test JSON format value."""
        assert ExportFormat.JSON.value == "json"

    def test_prometheus_value(self) -> None:
        """Test PROMETHEUS format value."""
        assert ExportFormat.PROMETHEUS.value == "prometheus"

    def test_otlp_value(self) -> None:
        """Test OTLP format value."""
        assert ExportFormat.OTLP.value == "otlp"

    def test_cloudwatch_value(self) -> None:
        """Test CLOUDWATCH format value."""
        assert ExportFormat.CLOUDWATCH.value == "cloudwatch"


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating LogConfig instance."""
        config = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        assert config.level == LogLevel.INFO
        assert config.format == "{message}"
        assert config.include_timestamp is True
        assert config.include_context is False

    def test_frozen(self) -> None:
        """Test that LogConfig is immutable."""
        config = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        with pytest.raises(AttributeError):
            config.level = LogLevel.DEBUG  # type: ignore[misc]


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating MetricConfig instance."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="requests_total",
            description="Total requests",
            labels=("method", "path"),
            buckets=None,
        )
        assert config.metric_type == MetricType.COUNTER
        assert config.name == "requests_total"
        assert config.labels == ("method", "path")

    def test_with_buckets(self) -> None:
        """Test MetricConfig with buckets for histogram."""
        config = MetricConfig(
            metric_type=MetricType.HISTOGRAM,
            name="latency",
            description="Latency",
            labels=(),
            buckets=(0.1, 0.5, 1.0),
        )
        assert config.buckets == (0.1, 0.5, 1.0)

    def test_frozen(self) -> None:
        """Test that MetricConfig is immutable."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="test",
            description="",
            labels=(),
            buckets=None,
        )
        with pytest.raises(AttributeError):
            config.name = "new_name"  # type: ignore[misc]


class TestTelemetryConfig:
    """Tests for TelemetryConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating TelemetryConfig instance."""
        log_cfg = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        config = TelemetryConfig(
            log_config=log_cfg,
            metrics=(),
            export_format=ExportFormat.JSON,
            batch_size=100,
            flush_interval=30.0,
        )
        assert config.log_config.level == LogLevel.INFO
        assert config.batch_size == 100
        assert config.flush_interval == pytest.approx(30.0)

    def test_frozen(self) -> None:
        """Test that TelemetryConfig is immutable."""
        log_cfg = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        config = TelemetryConfig(
            log_config=log_cfg,
            metrics=(),
            export_format=ExportFormat.JSON,
            batch_size=100,
            flush_interval=30.0,
        )
        with pytest.raises(AttributeError):
            config.batch_size = 200  # type: ignore[misc]


class TestTelemetryStats:
    """Tests for TelemetryStats dataclass."""

    def test_creation(self) -> None:
        """Test creating TelemetryStats instance."""
        stats = TelemetryStats(
            total_events=1000,
            events_by_level={"info": 800, "error": 200},
            total_metrics=5000,
            export_errors=2,
        )
        assert stats.total_events == 1000
        assert stats.events_by_level["info"] == 800
        assert stats.total_metrics == 5000
        assert stats.export_errors == 2

    def test_frozen(self) -> None:
        """Test that TelemetryStats is immutable."""
        stats = TelemetryStats(
            total_events=100,
            events_by_level={},
            total_metrics=0,
            export_errors=0,
        )
        with pytest.raises(AttributeError):
            stats.total_events = 200  # type: ignore[misc]


class TestValidateLogConfig:
    """Tests for validate_log_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        validate_log_config(config)  # Should not raise

    def test_empty_format_raises(self) -> None:
        """Test that empty format raises ValueError."""
        config = LogConfig(
            level=LogLevel.INFO,
            format="",
            include_timestamp=True,
            include_context=False,
        )
        with pytest.raises(ValueError, match="format cannot be empty"):
            validate_log_config(config)

    def test_format_too_long_raises(self) -> None:
        """Test that format exceeding 256 chars raises ValueError."""
        config = LogConfig(
            level=LogLevel.INFO,
            format="x" * 257,
            include_timestamp=True,
            include_context=False,
        )
        with pytest.raises(ValueError, match="format cannot exceed 256 characters"):
            validate_log_config(config)


class TestValidateMetricConfig:
    """Tests for validate_metric_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="test",
            description="Test metric",
            labels=(),
            buckets=None,
        )
        validate_metric_config(config)  # Should not raise

    def test_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="",
            description="",
            labels=(),
            buckets=None,
        )
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_metric_config(config)

    def test_name_too_long_raises(self) -> None:
        """Test that name exceeding 128 chars raises ValueError."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="x" * 129,
            description="",
            labels=(),
            buckets=None,
        )
        with pytest.raises(ValueError, match="name cannot exceed 128 characters"):
            validate_metric_config(config)

    def test_description_too_long_raises(self) -> None:
        """Test that description exceeding 512 chars raises ValueError."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="test",
            description="x" * 513,
            labels=(),
            buckets=None,
        )
        with pytest.raises(ValueError, match="description cannot exceed 512"):
            validate_metric_config(config)

    def test_empty_label_raises(self) -> None:
        """Test that empty label raises ValueError."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="test",
            description="",
            labels=("method", ""),
            buckets=None,
        )
        with pytest.raises(ValueError, match="labels cannot contain empty strings"):
            validate_metric_config(config)

    def test_label_too_long_raises(self) -> None:
        """Test that label exceeding 64 chars raises ValueError."""
        config = MetricConfig(
            metric_type=MetricType.COUNTER,
            name="test",
            description="",
            labels=("x" * 65,),
            buckets=None,
        )
        with pytest.raises(ValueError, match="label cannot exceed 64 characters"):
            validate_metric_config(config)

    def test_empty_buckets_raises(self) -> None:
        """Test that empty buckets tuple raises ValueError."""
        config = MetricConfig(
            metric_type=MetricType.HISTOGRAM,
            name="test",
            description="",
            labels=(),
            buckets=(),
        )
        with pytest.raises(ValueError, match="buckets cannot be empty if specified"):
            validate_metric_config(config)

    def test_unsorted_buckets_raises(self) -> None:
        """Test that unsorted buckets raise ValueError."""
        config = MetricConfig(
            metric_type=MetricType.HISTOGRAM,
            name="test",
            description="",
            labels=(),
            buckets=(1.0, 0.5, 2.0),
        )
        with pytest.raises(ValueError, match="buckets must be sorted"):
            validate_metric_config(config)

    def test_histogram_without_buckets_raises(self) -> None:
        """Test that histogram without buckets raises ValueError."""
        config = MetricConfig(
            metric_type=MetricType.HISTOGRAM,
            name="test",
            description="",
            labels=(),
            buckets=None,
        )
        with pytest.raises(ValueError, match="histogram metric requires buckets"):
            validate_metric_config(config)


class TestValidateTelemetryConfig:
    """Tests for validate_telemetry_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        log_cfg = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        config = TelemetryConfig(
            log_config=log_cfg,
            metrics=(),
            export_format=ExportFormat.JSON,
            batch_size=100,
            flush_interval=30.0,
        )
        validate_telemetry_config(config)  # Should not raise

    def test_zero_batch_size_raises(self) -> None:
        """Test that zero batch_size raises ValueError."""
        log_cfg = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        config = TelemetryConfig(
            log_config=log_cfg,
            metrics=(),
            export_format=ExportFormat.JSON,
            batch_size=0,
            flush_interval=30.0,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_telemetry_config(config)

    def test_batch_size_too_large_raises(self) -> None:
        """Test that batch_size exceeding 10000 raises ValueError."""
        log_cfg = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        config = TelemetryConfig(
            log_config=log_cfg,
            metrics=(),
            export_format=ExportFormat.JSON,
            batch_size=10001,
            flush_interval=30.0,
        )
        with pytest.raises(ValueError, match="batch_size cannot exceed 10000"):
            validate_telemetry_config(config)

    def test_zero_flush_interval_raises(self) -> None:
        """Test that zero flush_interval raises ValueError."""
        log_cfg = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        config = TelemetryConfig(
            log_config=log_cfg,
            metrics=(),
            export_format=ExportFormat.JSON,
            batch_size=100,
            flush_interval=0.0,
        )
        with pytest.raises(ValueError, match="flush_interval must be positive"):
            validate_telemetry_config(config)

    def test_flush_interval_too_large_raises(self) -> None:
        """Test that flush_interval exceeding 3600 raises ValueError."""
        log_cfg = LogConfig(
            level=LogLevel.INFO,
            format="{message}",
            include_timestamp=True,
            include_context=False,
        )
        config = TelemetryConfig(
            log_config=log_cfg,
            metrics=(),
            export_format=ExportFormat.JSON,
            batch_size=100,
            flush_interval=3601.0,
        )
        with pytest.raises(ValueError, match="flush_interval cannot exceed 3600"):
            validate_telemetry_config(config)


class TestValidateTelemetryStats:
    """Tests for validate_telemetry_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = TelemetryStats(
            total_events=100,
            events_by_level={"info": 80, "error": 20},
            total_metrics=500,
            export_errors=0,
        )
        validate_telemetry_stats(stats)  # Should not raise

    def test_negative_total_events_raises(self) -> None:
        """Test that negative total_events raises ValueError."""
        stats = TelemetryStats(
            total_events=-1,
            events_by_level={},
            total_metrics=0,
            export_errors=0,
        )
        with pytest.raises(ValueError, match="total_events cannot be negative"):
            validate_telemetry_stats(stats)

    def test_negative_total_metrics_raises(self) -> None:
        """Test that negative total_metrics raises ValueError."""
        stats = TelemetryStats(
            total_events=0,
            events_by_level={},
            total_metrics=-1,
            export_errors=0,
        )
        with pytest.raises(ValueError, match="total_metrics cannot be negative"):
            validate_telemetry_stats(stats)

    def test_negative_export_errors_raises(self) -> None:
        """Test that negative export_errors raises ValueError."""
        stats = TelemetryStats(
            total_events=0,
            events_by_level={},
            total_metrics=0,
            export_errors=-1,
        )
        with pytest.raises(ValueError, match="export_errors cannot be negative"):
            validate_telemetry_stats(stats)

    def test_invalid_level_in_events_by_level_raises(self) -> None:
        """Test that invalid level in events_by_level raises ValueError."""
        stats = TelemetryStats(
            total_events=100,
            events_by_level={"invalid": 100},
            total_metrics=0,
            export_errors=0,
        )
        with pytest.raises(ValueError, match="contains invalid level"):
            validate_telemetry_stats(stats)

    def test_negative_count_in_events_by_level_raises(self) -> None:
        """Test that negative count in events_by_level raises ValueError."""
        stats = TelemetryStats(
            total_events=100,
            events_by_level={"info": -10},
            total_metrics=0,
            export_errors=0,
        )
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_telemetry_stats(stats)


class TestCreateLogConfig:
    """Tests for create_log_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_log_config()
        assert config.level == LogLevel.INFO
        assert config.include_timestamp is True
        assert config.include_context is True

    def test_custom_level(self) -> None:
        """Test creation with custom level."""
        config = create_log_config(level="debug")
        assert config.level == LogLevel.DEBUG

    def test_custom_format(self) -> None:
        """Test creation with custom format."""
        config = create_log_config(format="{level}: {message}")
        assert config.format == "{level}: {message}"

    def test_invalid_level_raises(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="level must be one of"):
            create_log_config(level="invalid")


class TestCreateMetricConfig:
    """Tests for create_metric_config function."""

    def test_counter_creation(self) -> None:
        """Test creating counter metric."""
        config = create_metric_config("counter", "requests_total")
        assert config.metric_type == MetricType.COUNTER
        assert config.name == "requests_total"

    def test_histogram_with_buckets(self) -> None:
        """Test creating histogram with buckets."""
        config = create_metric_config(
            "histogram",
            "latency",
            buckets=[0.1, 0.5, 1.0],
        )
        assert config.metric_type == MetricType.HISTOGRAM
        assert config.buckets == (0.1, 0.5, 1.0)

    def test_with_labels(self) -> None:
        """Test creating metric with labels."""
        config = create_metric_config(
            "counter",
            "requests",
            labels=["method", "path"],
        )
        assert config.labels == ("method", "path")

    def test_invalid_metric_type_raises(self) -> None:
        """Test that invalid metric_type raises ValueError."""
        with pytest.raises(ValueError, match="metric_type must be one of"):
            create_metric_config("invalid", "test")


class TestCreateTelemetryConfig:
    """Tests for create_telemetry_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_telemetry_config()
        assert config.export_format == ExportFormat.JSON
        assert config.batch_size == 100
        assert config.flush_interval == pytest.approx(30.0)

    def test_with_log_config(self) -> None:
        """Test creation with custom log config."""
        log_cfg = create_log_config(level="debug")
        config = create_telemetry_config(log_config=log_cfg)
        assert config.log_config.level == LogLevel.DEBUG

    def test_with_metrics(self) -> None:
        """Test creation with metrics."""
        metric = create_metric_config("counter", "test")
        config = create_telemetry_config(metrics=[metric])
        assert len(config.metrics) == 1

    def test_invalid_export_format_raises(self) -> None:
        """Test that invalid export_format raises ValueError."""
        with pytest.raises(ValueError, match="export_format must be one of"):
            create_telemetry_config(export_format="invalid")


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_log_levels_returns_sorted(self) -> None:
        """Test that list_log_levels returns sorted list."""
        levels = list_log_levels()
        assert levels == sorted(levels)
        assert "info" in levels
        assert "error" in levels

    def test_list_metric_types_returns_sorted(self) -> None:
        """Test that list_metric_types returns sorted list."""
        types = list_metric_types()
        assert types == sorted(types)
        assert "counter" in types
        assert "histogram" in types

    def test_list_export_formats_returns_sorted(self) -> None:
        """Test that list_export_formats returns sorted list."""
        formats = list_export_formats()
        assert formats == sorted(formats)
        assert "json" in formats
        assert "prometheus" in formats


class TestGetFunctions:
    """Tests for get_* functions."""

    def test_get_log_level_valid(self) -> None:
        """Test get_log_level with valid name."""
        assert get_log_level("info") == LogLevel.INFO
        assert get_log_level("debug") == LogLevel.DEBUG

    def test_get_log_level_invalid_raises(self) -> None:
        """Test get_log_level with invalid name raises."""
        with pytest.raises(ValueError, match="level must be one of"):
            get_log_level("invalid")

    def test_get_metric_type_valid(self) -> None:
        """Test get_metric_type with valid name."""
        assert get_metric_type("counter") == MetricType.COUNTER
        assert get_metric_type("histogram") == MetricType.HISTOGRAM

    def test_get_metric_type_invalid_raises(self) -> None:
        """Test get_metric_type with invalid name raises."""
        with pytest.raises(ValueError, match="metric_type must be one of"):
            get_metric_type("invalid")

    def test_get_export_format_valid(self) -> None:
        """Test get_export_format with valid name."""
        assert get_export_format("json") == ExportFormat.JSON
        assert get_export_format("prometheus") == ExportFormat.PROMETHEUS

    def test_get_export_format_invalid_raises(self) -> None:
        """Test get_export_format with invalid name raises."""
        with pytest.raises(ValueError, match="export_format must be one of"):
            get_export_format("invalid")


class TestFormatLogMessage:
    """Tests for format_log_message function."""

    def test_basic_message(self) -> None:
        """Test formatting basic message."""
        msg = format_log_message("Hello, world!")
        assert "[info]" in msg
        assert "Hello, world!" in msg

    def test_with_level(self) -> None:
        """Test formatting with custom level."""
        msg = format_log_message("Error occurred", level="error")
        assert "[error]" in msg

    def test_with_context(self) -> None:
        """Test formatting with context."""
        msg = format_log_message(
            "User action",
            context={"user_id": "123", "action": "login"},
        )
        assert "user_id=123" in msg
        assert "action=login" in msg

    def test_without_timestamp(self) -> None:
        """Test formatting without timestamp."""
        config = create_log_config(include_timestamp=False)
        msg = format_log_message("Test", config=config)
        assert "[info]" in msg
        assert "Test" in msg

    def test_without_context(self) -> None:
        """Test formatting without context in config."""
        config = create_log_config(include_context=False)
        msg = format_log_message(
            "Test",
            context={"key": "value"},
            config=config,
        )
        assert "key=value" not in msg

    def test_invalid_level_raises(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="level must be one of"):
            format_log_message("Test", level="invalid")


class TestRecordMetric:
    """Tests for record_metric function."""

    def test_basic_recording(self) -> None:
        """Test basic metric recording."""
        config = create_metric_config("counter", "requests")
        val = record_metric(config, 1.0)
        assert val.value == pytest.approx(1.0)
        assert val.config.name == "requests"

    def test_with_labels(self) -> None:
        """Test recording with labels."""
        config = create_metric_config(
            "counter",
            "requests",
            labels=["method"],
        )
        val = record_metric(config, 1.0, labels={"method": "GET"})
        assert val.labels["method"] == "GET"

    def test_missing_labels_raises(self) -> None:
        """Test that missing labels raises ValueError."""
        config = create_metric_config(
            "counter",
            "requests",
            labels=["method"],
        )
        with pytest.raises(ValueError, match="missing labels"):
            record_metric(config, 1.0)

    def test_extra_labels_raises(self) -> None:
        """Test that extra labels raises ValueError."""
        config = create_metric_config("counter", "requests")
        with pytest.raises(ValueError, match="extra labels"):
            record_metric(config, 1.0, labels={"extra": "value"})

    def test_negative_counter_raises(self) -> None:
        """Test that negative counter value raises ValueError."""
        config = create_metric_config("counter", "count")
        with pytest.raises(ValueError, match="counter metric value cannot be negative"):
            record_metric(config, -1.0)

    def test_negative_gauge_allowed(self) -> None:
        """Test that negative gauge value is allowed."""
        config = create_metric_config("gauge", "temperature")
        val = record_metric(config, -10.0)
        assert val.value == -10.0


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_empty_list(self) -> None:
        """Test aggregation of empty list."""
        result = aggregate_metrics([])
        assert result == {}

    def test_counter_aggregation(self) -> None:
        """Test counter values are summed."""
        config = create_metric_config("counter", "requests")
        vals = [
            record_metric(config, 1.0),
            record_metric(config, 2.0),
            record_metric(config, 3.0),
        ]
        result = aggregate_metrics(vals)
        assert result["requests"]["_total"] == pytest.approx(6.0)

    def test_gauge_aggregation(self) -> None:
        """Test gauge values use last value."""
        config = create_metric_config("gauge", "temperature")
        vals = [
            record_metric(config, 20.0),
            record_metric(config, 25.0),
            record_metric(config, 22.0),
        ]
        result = aggregate_metrics(vals)
        assert result["temperature"]["_total"] == pytest.approx(22.0)

    def test_with_labels(self) -> None:
        """Test aggregation with labels."""
        config = create_metric_config("counter", "requests", labels=["method"])
        vals = [
            record_metric(config, 1.0, {"method": "GET"}),
            record_metric(config, 2.0, {"method": "GET"}),
            record_metric(config, 1.0, {"method": "POST"}),
        ]
        result = aggregate_metrics(vals)
        assert result["requests"]["method=GET"] == 3.0
        assert result["requests"]["method=POST"] == 1.0


class TestExportTelemetry:
    """Tests for export_telemetry function."""

    def test_json_export(self) -> None:
        """Test JSON export format."""
        config = create_metric_config("counter", "requests")
        vals = [record_metric(config, 5.0)]
        output = export_telemetry(vals, "json")
        data = json.loads(output)
        assert "metrics" in data
        assert "requests" in data["metrics"]

    def test_prometheus_export(self) -> None:
        """Test Prometheus export format."""
        config = create_metric_config("counter", "requests_total", "Total requests")
        vals = [record_metric(config, 5.0)]
        output = export_telemetry(vals, "prometheus")
        assert "# HELP requests_total" in output
        assert "# TYPE requests_total counter" in output
        assert "requests_total 5.0" in output

    def test_prometheus_with_labels(self) -> None:
        """Test Prometheus export with labels."""
        config = create_metric_config("counter", "requests", labels=["method"])
        vals = [record_metric(config, 3.0, {"method": "GET"})]
        output = export_telemetry(vals, "prometheus")
        assert 'requests{method="GET"}' in output

    def test_otlp_export(self) -> None:
        """Test OTLP export format."""
        config = create_metric_config("counter", "requests")
        vals = [record_metric(config, 5.0)]
        output = export_telemetry(vals, "otlp")
        data = json.loads(output)
        assert "resourceMetrics" in data

    def test_cloudwatch_export(self) -> None:
        """Test CloudWatch export format."""
        config = create_metric_config("counter", "requests")
        vals = [record_metric(config, 5.0)]
        output = export_telemetry(vals, "cloudwatch")
        data = json.loads(output)
        assert "MetricData" in data
        assert "Namespace" in data

    def test_invalid_format_raises(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="export_format must be one of"):
            export_telemetry([], "invalid")


class TestCalculatePercentiles:
    """Tests for calculate_percentiles function."""

    def test_basic_percentiles(self) -> None:
        """Test basic percentile calculation."""
        vals = list(range(1, 11))
        result = calculate_percentiles(vals, [50.0])
        assert result[50.0] == pytest.approx(5.5)

    def test_default_percentiles(self) -> None:
        """Test default percentile values."""
        vals = list(range(1, 101))
        result = calculate_percentiles(vals)
        assert 50.0 in result
        assert 90.0 in result
        assert 95.0 in result
        assert 99.0 in result

    def test_empty_list_raises(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="values cannot be empty"):
            calculate_percentiles([])

    def test_invalid_percentile_raises(self) -> None:
        """Test that invalid percentile raises ValueError."""
        with pytest.raises(ValueError, match="percentile must be between 0 and 100"):
            calculate_percentiles([1, 2, 3], [101.0])

    def test_negative_percentile_raises(self) -> None:
        """Test that negative percentile raises ValueError."""
        with pytest.raises(ValueError, match="percentile must be between 0 and 100"):
            calculate_percentiles([1, 2, 3], [-1.0])

    def test_single_value(self) -> None:
        """Test percentiles with single value."""
        result = calculate_percentiles([42.0], [50.0])
        assert result[50.0] == pytest.approx(42.0)


class TestFormatTelemetryStats:
    """Tests for format_telemetry_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = TelemetryStats(
            total_events=1000,
            events_by_level={"info": 800, "error": 200},
            total_metrics=5000,
            export_errors=2,
        )
        output = format_telemetry_stats(stats)
        assert "Total Events: 1000" in output
        assert "Total Metrics: 5000" in output
        assert "Export Errors: 2" in output
        assert "info: 800" in output
        assert "error: 200" in output

    def test_empty_events_by_level(self) -> None:
        """Test formatting with empty events_by_level."""
        stats = TelemetryStats(
            total_events=0,
            events_by_level={},
            total_metrics=0,
            export_errors=0,
        )
        output = format_telemetry_stats(stats)
        assert "Total Events: 0" in output


class TestGetRecommendedTelemetryConfig:
    """Tests for get_recommended_telemetry_config function."""

    def test_development_config(self) -> None:
        """Test development configuration."""
        config = get_recommended_telemetry_config("development")
        assert config.log_config.level == LogLevel.DEBUG
        assert config.batch_size == 10
        assert config.flush_interval == pytest.approx(5.0)
        assert config.export_format == ExportFormat.JSON

    def test_production_config(self) -> None:
        """Test production configuration."""
        config = get_recommended_telemetry_config("production")
        assert config.log_config.level == LogLevel.INFO
        assert config.batch_size == 100
        assert config.flush_interval == pytest.approx(30.0)
        assert config.export_format == ExportFormat.PROMETHEUS

    def test_debugging_config(self) -> None:
        """Test debugging configuration."""
        config = get_recommended_telemetry_config("debugging")
        assert config.log_config.level == LogLevel.DEBUG
        assert config.log_config.include_context is True
        assert config.batch_size == 1
        assert config.flush_interval == pytest.approx(1.0)

    def test_invalid_use_case_raises(self) -> None:
        """Test that invalid use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_telemetry_config("invalid")

    def test_development_has_metrics(self) -> None:
        """Test that development config has metrics."""
        config = get_recommended_telemetry_config("development")
        assert len(config.metrics) >= 2
        metric_names = [m.name for m in config.metrics]
        assert "requests_total" in metric_names

    def test_production_has_metrics(self) -> None:
        """Test that production config has metrics."""
        config = get_recommended_telemetry_config("production")
        assert len(config.metrics) >= 4
        metric_names = [m.name for m in config.metrics]
        assert "errors_total" in metric_names

    def test_debugging_has_metrics(self) -> None:
        """Test that debugging config has metrics."""
        config = get_recommended_telemetry_config("debugging")
        assert len(config.metrics) >= 3
        metric_names = [m.name for m in config.metrics]
        assert "memory_usage" in metric_names


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_creation(self) -> None:
        """Test creating MetricValue instance."""
        config = create_metric_config("counter", "test")
        val = MetricValue(
            config=config,
            value=1.0,
            labels={},
            timestamp=datetime.now(UTC),
        )
        assert val.value == pytest.approx(1.0)
        assert val.config.name == "test"

    def test_frozen(self) -> None:
        """Test that MetricValue is immutable."""
        config = create_metric_config("counter", "test")
        val = MetricValue(
            config=config,
            value=1.0,
            labels={},
            timestamp=datetime.now(UTC),
        )
        with pytest.raises(AttributeError):
            val.value = 2.0  # type: ignore[misc]


class TestValidConstants:
    """Tests for VALID_* constants."""

    def test_valid_log_levels(self) -> None:
        """Test VALID_LOG_LEVELS contains all levels."""
        assert (
            frozenset({"debug", "info", "warning", "error", "critical"})
            == VALID_LOG_LEVELS
        )

    def test_valid_metric_types(self) -> None:
        """Test VALID_METRIC_TYPES contains all types."""
        assert (
            frozenset({"counter", "gauge", "histogram", "summary"})
            == VALID_METRIC_TYPES
        )

    def test_valid_export_formats(self) -> None:
        """Test VALID_EXPORT_FORMATS contains all formats."""
        assert (
            frozenset({"json", "prometheus", "otlp", "cloudwatch"})
            == VALID_EXPORT_FORMATS
        )
