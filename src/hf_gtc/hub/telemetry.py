"""Telemetry, logging, and usage tracking utilities for HuggingFace Hub.

This module provides dataclasses and functions for logging, metrics collection,
and telemetry export for ML applications using the HuggingFace Hub.

Examples:
    >>> from hf_gtc.hub.telemetry import create_log_config, LogLevel
    >>> config = create_log_config(level="info")
    >>> config.level
    <LogLevel.INFO: 'info'>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class LogLevel(Enum):
    """Log level for telemetry logging.

    Attributes:
        DEBUG: Detailed debugging information.
        INFO: General informational messages.
        WARNING: Warning messages for potential issues.
        ERROR: Error messages for failures.
        CRITICAL: Critical errors that may cause termination.

    Examples:
        >>> LogLevel.DEBUG.value
        'debug'
        >>> LogLevel.INFO.value
        'info'
        >>> LogLevel.WARNING.value
        'warning'
        >>> LogLevel.ERROR.value
        'error'
        >>> LogLevel.CRITICAL.value
        'critical'
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


VALID_LOG_LEVELS = frozenset(level.value for level in LogLevel)


class MetricType(Enum):
    """Type of metric for telemetry.

    Attributes:
        COUNTER: Monotonically increasing counter.
        GAUGE: Value that can go up or down.
        HISTOGRAM: Distribution of values with buckets.
        SUMMARY: Similar to histogram but calculates quantiles.

    Examples:
        >>> MetricType.COUNTER.value
        'counter'
        >>> MetricType.GAUGE.value
        'gauge'
        >>> MetricType.HISTOGRAM.value
        'histogram'
        >>> MetricType.SUMMARY.value
        'summary'
    """

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


VALID_METRIC_TYPES = frozenset(mt.value for mt in MetricType)


class ExportFormat(Enum):
    """Export format for telemetry data.

    Attributes:
        JSON: JSON format for general use.
        PROMETHEUS: Prometheus text format.
        OTLP: OpenTelemetry Protocol format.
        CLOUDWATCH: AWS CloudWatch format.

    Examples:
        >>> ExportFormat.JSON.value
        'json'
        >>> ExportFormat.PROMETHEUS.value
        'prometheus'
        >>> ExportFormat.OTLP.value
        'otlp'
        >>> ExportFormat.CLOUDWATCH.value
        'cloudwatch'
    """

    JSON = "json"
    PROMETHEUS = "prometheus"
    OTLP = "otlp"
    CLOUDWATCH = "cloudwatch"


VALID_EXPORT_FORMATS = frozenset(ef.value for ef in ExportFormat)


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Configuration for log messages.

    Attributes:
        level: Minimum log level to capture.
        format: Log message format string.
        include_timestamp: Whether to include timestamp in logs.
        include_context: Whether to include context metadata.

    Examples:
        >>> config = LogConfig(
        ...     level=LogLevel.INFO,
        ...     format="{timestamp} [{level}] {message}",
        ...     include_timestamp=True,
        ...     include_context=True,
        ... )
        >>> config.level
        <LogLevel.INFO: 'info'>
        >>> config.include_timestamp
        True
    """

    level: LogLevel
    format: str
    include_timestamp: bool
    include_context: bool


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Configuration for a tracked metric.

    Attributes:
        metric_type: Type of the metric.
        name: Metric name.
        description: Human-readable description.
        labels: Labels for metric dimensions.
        buckets: Bucket boundaries for histograms.

    Examples:
        >>> config = MetricConfig(
        ...     metric_type=MetricType.COUNTER,
        ...     name="requests_total",
        ...     description="Total number of requests",
        ...     labels=("method", "endpoint"),
        ...     buckets=None,
        ... )
        >>> config.name
        'requests_total'
        >>> config.labels
        ('method', 'endpoint')
    """

    metric_type: MetricType
    name: str
    description: str
    labels: tuple[str, ...]
    buckets: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Configuration for telemetry collection and export.

    Attributes:
        log_config: Configuration for logging.
        metrics: Tuple of metric configurations.
        export_format: Format for exporting telemetry.
        batch_size: Number of events to batch before export.
        flush_interval: Seconds between automatic flushes.

    Examples:
        >>> log_cfg = LogConfig(
        ...     level=LogLevel.INFO,
        ...     format="{message}",
        ...     include_timestamp=True,
        ...     include_context=False,
        ... )
        >>> config = TelemetryConfig(
        ...     log_config=log_cfg,
        ...     metrics=(),
        ...     export_format=ExportFormat.JSON,
        ...     batch_size=100,
        ...     flush_interval=30.0,
        ... )
        >>> config.batch_size
        100
        >>> config.flush_interval
        30.0
    """

    log_config: LogConfig
    metrics: tuple[MetricConfig, ...]
    export_format: ExportFormat
    batch_size: int
    flush_interval: float


@dataclass(frozen=True, slots=True)
class TelemetryStats:
    """Statistics for telemetry collection.

    Attributes:
        total_events: Total number of events logged.
        events_by_level: Count of events per log level.
        total_metrics: Total number of metric recordings.
        export_errors: Number of export errors.

    Examples:
        >>> stats = TelemetryStats(
        ...     total_events=1000,
        ...     events_by_level={"info": 800, "error": 200},
        ...     total_metrics=5000,
        ...     export_errors=2,
        ... )
        >>> stats.total_events
        1000
        >>> stats.events_by_level["info"]
        800
    """

    total_events: int
    events_by_level: dict[str, int]
    total_metrics: int
    export_errors: int


def validate_log_config(config: LogConfig) -> None:
    """Validate log configuration.

    Args:
        config: Log configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = LogConfig(
        ...     level=LogLevel.INFO,
        ...     format="{message}",
        ...     include_timestamp=True,
        ...     include_context=False,
        ... )
        >>> validate_log_config(config)  # No error

        >>> bad = LogConfig(
        ...     level=LogLevel.INFO,
        ...     format="",
        ...     include_timestamp=True,
        ...     include_context=False,
        ... )
        >>> validate_log_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: format cannot be empty
    """
    if not config.format:
        msg = "format cannot be empty"
        raise ValueError(msg)

    if len(config.format) > 256:
        msg = f"format cannot exceed 256 characters, got {len(config.format)}"
        raise ValueError(msg)


def validate_metric_config(config: MetricConfig) -> None:
    """Validate metric configuration.

    Args:
        config: Metric configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = MetricConfig(
        ...     metric_type=MetricType.COUNTER,
        ...     name="requests",
        ...     description="Request count",
        ...     labels=(),
        ...     buckets=None,
        ... )
        >>> validate_metric_config(config)  # No error

        >>> bad = MetricConfig(
        ...     metric_type=MetricType.COUNTER,
        ...     name="",
        ...     description="",
        ...     labels=(),
        ...     buckets=None,
        ... )
        >>> validate_metric_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if not config.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if len(config.name) > 128:
        msg = f"name cannot exceed 128 characters, got {len(config.name)}"
        raise ValueError(msg)

    if len(config.description) > 512:
        msg = f"description cannot exceed 512 characters, got {len(config.description)}"
        raise ValueError(msg)

    _validate_metric_labels(config.labels)
    _validate_metric_buckets(config)


def _validate_metric_labels(labels: tuple[str, ...]) -> None:
    """Validate metric labels."""
    for label in labels:
        if not label:
            msg = "labels cannot contain empty strings"
            raise ValueError(msg)
        if len(label) > 64:
            msg = f"label cannot exceed 64 characters, got '{label}'"
            raise ValueError(msg)


def _validate_metric_buckets(config: MetricConfig) -> None:
    """Validate metric buckets."""
    if config.buckets is not None:
        if len(config.buckets) == 0:
            msg = "buckets cannot be empty if specified"
            raise ValueError(msg)
        sorted_buckets = tuple(sorted(config.buckets))
        if config.buckets != sorted_buckets:
            msg = "buckets must be sorted in ascending order"
            raise ValueError(msg)

    if config.metric_type == MetricType.HISTOGRAM and config.buckets is None:
        msg = "histogram metric requires buckets to be specified"
        raise ValueError(msg)


def validate_telemetry_config(config: TelemetryConfig) -> None:
    """Validate telemetry configuration.

    Args:
        config: Telemetry configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> log_cfg = LogConfig(
        ...     level=LogLevel.INFO,
        ...     format="{message}",
        ...     include_timestamp=True,
        ...     include_context=False,
        ... )
        >>> config = TelemetryConfig(
        ...     log_config=log_cfg,
        ...     metrics=(),
        ...     export_format=ExportFormat.JSON,
        ...     batch_size=100,
        ...     flush_interval=30.0,
        ... )
        >>> validate_telemetry_config(config)  # No error

        >>> bad = TelemetryConfig(
        ...     log_config=log_cfg,
        ...     metrics=(),
        ...     export_format=ExportFormat.JSON,
        ...     batch_size=0,
        ...     flush_interval=30.0,
        ... )
        >>> validate_telemetry_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    validate_log_config(config.log_config)

    for metric in config.metrics:
        validate_metric_config(metric)

    if config.batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if config.batch_size > 10000:
        msg = f"batch_size cannot exceed 10000, got {config.batch_size}"
        raise ValueError(msg)

    if config.flush_interval <= 0:
        msg = "flush_interval must be positive"
        raise ValueError(msg)

    if config.flush_interval > 3600:
        msg = f"flush_interval cannot exceed 3600 seconds, got {config.flush_interval}"
        raise ValueError(msg)


def validate_telemetry_stats(stats: TelemetryStats) -> None:
    """Validate telemetry stats.

    Args:
        stats: Telemetry stats to validate.

    Raises:
        ValueError: If stats are invalid.

    Examples:
        >>> stats = TelemetryStats(
        ...     total_events=100,
        ...     events_by_level={"info": 80, "error": 20},
        ...     total_metrics=500,
        ...     export_errors=0,
        ... )
        >>> validate_telemetry_stats(stats)  # No error

        >>> bad = TelemetryStats(
        ...     total_events=-1,
        ...     events_by_level={},
        ...     total_metrics=0,
        ...     export_errors=0,
        ... )
        >>> validate_telemetry_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_events cannot be negative
    """
    if stats.total_events < 0:
        msg = "total_events cannot be negative"
        raise ValueError(msg)

    if stats.total_metrics < 0:
        msg = "total_metrics cannot be negative"
        raise ValueError(msg)

    if stats.export_errors < 0:
        msg = "export_errors cannot be negative"
        raise ValueError(msg)

    # Validate events_by_level keys are valid log levels
    for level in stats.events_by_level:
        if level not in VALID_LOG_LEVELS:
            msg = f"events_by_level contains invalid level '{level}'"
            raise ValueError(msg)

    # Validate counts are non-negative
    for level, count in stats.events_by_level.items():
        if count < 0:
            msg = f"events_by_level['{level}'] cannot be negative"
            raise ValueError(msg)


def create_log_config(
    level: str = "info",
    format: str = "{timestamp} [{level}] {message}",
    include_timestamp: bool = True,
    include_context: bool = True,
) -> LogConfig:
    """Create a log configuration.

    Args:
        level: Log level string. Defaults to "info".
        format: Log message format. Defaults to "{timestamp} [{level}] {message}".
        include_timestamp: Include timestamp. Defaults to True.
        include_context: Include context. Defaults to True.

    Returns:
        LogConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_log_config()
        >>> config.level
        <LogLevel.INFO: 'info'>
        >>> config.include_timestamp
        True

        >>> config = create_log_config(level="debug", include_context=False)
        >>> config.level
        <LogLevel.DEBUG: 'debug'>
        >>> config.include_context
        False

        >>> create_log_config(level="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: level must be one of
    """
    if level not in VALID_LOG_LEVELS:
        msg = f"level must be one of {VALID_LOG_LEVELS}, got '{level}'"
        raise ValueError(msg)

    config = LogConfig(
        level=LogLevel(level),
        format=format,
        include_timestamp=include_timestamp,
        include_context=include_context,
    )
    validate_log_config(config)
    return config


def create_metric_config(
    metric_type: str,
    name: str,
    description: str = "",
    labels: tuple[str, ...] | list[str] | None = None,
    buckets: tuple[float, ...] | list[float] | None = None,
) -> MetricConfig:
    """Create a metric configuration.

    Args:
        metric_type: Type of metric.
        name: Metric name.
        description: Human-readable description. Defaults to "".
        labels: Labels for dimensions. Defaults to None.
        buckets: Bucket boundaries for histograms. Defaults to None.

    Returns:
        MetricConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_metric_config("counter", "requests_total")
        >>> config.name
        'requests_total'
        >>> config.metric_type
        <MetricType.COUNTER: 'counter'>

        >>> config = create_metric_config(
        ...     "histogram",
        ...     "latency",
        ...     labels=["method"],
        ...     buckets=[0.1, 0.5, 1.0, 5.0],
        ... )
        >>> config.labels
        ('method',)
        >>> config.buckets
        (0.1, 0.5, 1.0, 5.0)

        >>> create_metric_config("invalid", "x")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric_type must be one of
    """
    if metric_type not in VALID_METRIC_TYPES:
        msg = f"metric_type must be one of {VALID_METRIC_TYPES}, got '{metric_type}'"
        raise ValueError(msg)

    labels_tuple = tuple(labels) if labels is not None else ()
    buckets_tuple = tuple(buckets) if buckets is not None else None

    config = MetricConfig(
        metric_type=MetricType(metric_type),
        name=name,
        description=description,
        labels=labels_tuple,
        buckets=buckets_tuple,
    )
    validate_metric_config(config)
    return config


def create_telemetry_config(
    log_config: LogConfig | None = None,
    metrics: tuple[MetricConfig, ...] | list[MetricConfig] | None = None,
    export_format: str = "json",
    batch_size: int = 100,
    flush_interval: float = 30.0,
) -> TelemetryConfig:
    """Create a telemetry configuration.

    Args:
        log_config: Log configuration. Defaults to None (uses default).
        metrics: Metric configurations. Defaults to None.
        export_format: Export format string. Defaults to "json".
        batch_size: Batch size for export. Defaults to 100.
        flush_interval: Flush interval in seconds. Defaults to 30.0.

    Returns:
        TelemetryConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_telemetry_config()
        >>> config.export_format
        <ExportFormat.JSON: 'json'>
        >>> config.batch_size
        100

        >>> log_cfg = create_log_config(level="debug")
        >>> config = create_telemetry_config(
        ...     log_config=log_cfg,
        ...     export_format="prometheus",
        ...     batch_size=50,
        ... )
        >>> config.log_config.level
        <LogLevel.DEBUG: 'debug'>
        >>> config.export_format
        <ExportFormat.PROMETHEUS: 'prometheus'>

        >>> create_telemetry_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     export_format="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: export_format must be one of
    """
    if export_format not in VALID_EXPORT_FORMATS:
        msg = f"export_format must be one of {VALID_EXPORT_FORMATS}, got '{export_format}'"  # noqa: E501
        raise ValueError(msg)

    if log_config is None:
        log_config = create_log_config()

    metrics_tuple = tuple(metrics) if metrics is not None else ()

    config = TelemetryConfig(
        log_config=log_config,
        metrics=metrics_tuple,
        export_format=ExportFormat(export_format),
        batch_size=batch_size,
        flush_interval=flush_interval,
    )
    validate_telemetry_config(config)
    return config


def list_log_levels() -> list[str]:
    """List all valid log levels.

    Returns:
        Sorted list of log level names.

    Examples:
        >>> levels = list_log_levels()
        >>> "info" in levels
        True
        >>> "error" in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_LOG_LEVELS)


def list_metric_types() -> list[str]:
    """List all valid metric types.

    Returns:
        Sorted list of metric type names.

    Examples:
        >>> types = list_metric_types()
        >>> "counter" in types
        True
        >>> "histogram" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_METRIC_TYPES)


def list_export_formats() -> list[str]:
    """List all valid export formats.

    Returns:
        Sorted list of export format names.

    Examples:
        >>> formats = list_export_formats()
        >>> "json" in formats
        True
        >>> "prometheus" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_EXPORT_FORMATS)


def get_log_level(name: str) -> LogLevel:
    """Get log level from string name.

    Args:
        name: Log level name.

    Returns:
        LogLevel enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_log_level("info")
        <LogLevel.INFO: 'info'>

        >>> get_log_level("debug")
        <LogLevel.DEBUG: 'debug'>

        >>> get_log_level("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: level must be one of
    """
    if name not in VALID_LOG_LEVELS:
        msg = f"level must be one of {VALID_LOG_LEVELS}, got '{name}'"
        raise ValueError(msg)
    return LogLevel(name)


def get_metric_type(name: str) -> MetricType:
    """Get metric type from string name.

    Args:
        name: Metric type name.

    Returns:
        MetricType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_metric_type("counter")
        <MetricType.COUNTER: 'counter'>

        >>> get_metric_type("histogram")
        <MetricType.HISTOGRAM: 'histogram'>

        >>> get_metric_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric_type must be one of
    """
    if name not in VALID_METRIC_TYPES:
        msg = f"metric_type must be one of {VALID_METRIC_TYPES}, got '{name}'"
        raise ValueError(msg)
    return MetricType(name)


def get_export_format(name: str) -> ExportFormat:
    """Get export format from string name.

    Args:
        name: Export format name.

    Returns:
        ExportFormat enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_export_format("json")
        <ExportFormat.JSON: 'json'>

        >>> get_export_format("prometheus")
        <ExportFormat.PROMETHEUS: 'prometheus'>

        >>> get_export_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: export_format must be one of
    """
    if name not in VALID_EXPORT_FORMATS:
        msg = f"export_format must be one of {VALID_EXPORT_FORMATS}, got '{name}'"
        raise ValueError(msg)
    return ExportFormat(name)


def format_log_message(
    message: str,
    level: str = "info",
    context: dict[str, Any] | None = None,
    config: LogConfig | None = None,
) -> str:
    """Format a log message according to configuration.

    Args:
        message: The log message.
        level: Log level. Defaults to "info".
        context: Additional context metadata. Defaults to None.
        config: Log configuration. Defaults to None (uses default).

    Returns:
        Formatted log message string.

    Raises:
        ValueError: If level is invalid.

    Examples:
        >>> msg = format_log_message("Hello, world!")
        >>> "[info]" in msg
        True
        >>> "Hello, world!" in msg
        True

        >>> msg = format_log_message(
        ...     "Error occurred",
        ...     level="error",
        ...     context={"user_id": "123"},
        ... )
        >>> "[error]" in msg
        True
        >>> "user_id" in msg
        True

        >>> config = create_log_config(include_timestamp=False)
        >>> msg = format_log_message("Test", config=config)
        >>> len(msg) > 0
        True

        >>> format_log_message(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "x", level="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: level must be one of
    """
    if level not in VALID_LOG_LEVELS:
        msg = f"level must be one of {VALID_LOG_LEVELS}, got '{level}'"
        raise ValueError(msg)

    if config is None:
        config = create_log_config()

    parts = []

    if config.include_timestamp:
        timestamp = datetime.now(UTC).isoformat()
        parts.append(timestamp)

    parts.append(f"[{level}]")
    parts.append(message)

    if config.include_context and context:
        context_str = " ".join(f"{k}={v}" for k, v in sorted(context.items()))
        parts.append(f"({context_str})")

    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A recorded metric value.

    Attributes:
        config: Metric configuration.
        value: The metric value.
        labels: Label values for this recording.
        timestamp: When the metric was recorded.

    Examples:
        >>> config = create_metric_config("counter", "requests")
        >>> val = MetricValue(
        ...     config=config,
        ...     value=1.0,
        ...     labels={},
        ...     timestamp=datetime.now(UTC),
        ... )
        >>> val.value
        1.0
    """

    config: MetricConfig
    value: float
    labels: dict[str, str]
    timestamp: datetime


def record_metric(
    config: MetricConfig,
    value: float,
    labels: dict[str, str] | None = None,
) -> MetricValue:
    """Record a metric value.

    Args:
        config: Metric configuration.
        value: The metric value.
        labels: Label values. Defaults to None.

    Returns:
        MetricValue with the recorded data.

    Raises:
        ValueError: If labels don't match config or value is invalid.

    Examples:
        >>> config = create_metric_config("counter", "requests")
        >>> val = record_metric(config, 1.0)
        >>> val.value
        1.0

        >>> config = create_metric_config("gauge", "temperature", labels=["location"])
        >>> val = record_metric(config, 23.5, labels={"location": "server1"})
        >>> val.labels["location"]
        'server1'

        >>> config = create_metric_config("counter", "count")
        >>> record_metric(config, -1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: counter metric value cannot be negative
    """
    labels = labels or {}

    # Validate labels match config
    config_labels = set(config.labels)
    provided_labels = set(labels.keys())

    if config_labels != provided_labels:
        missing = config_labels - provided_labels
        extra = provided_labels - config_labels
        parts = []
        if missing:
            parts.append(f"missing labels: {missing}")
        if extra:
            parts.append(f"extra labels: {extra}")
        msg = "; ".join(parts)
        raise ValueError(msg)

    # Counter values cannot be negative
    if config.metric_type == MetricType.COUNTER and value < 0:
        msg = "counter metric value cannot be negative"
        raise ValueError(msg)

    return MetricValue(
        config=config,
        value=value,
        labels=labels,
        timestamp=datetime.now(UTC),
    )


def aggregate_metrics(
    values: list[MetricValue],
) -> dict[str, dict[str, float]]:
    """Aggregate metric values by name and labels.

    Args:
        values: List of metric values to aggregate.

    Returns:
        Dict mapping metric name to dict of label_key to aggregated value.

    Examples:
        >>> config = create_metric_config("counter", "requests")
        >>> vals = [
        ...     record_metric(config, 1.0),
        ...     record_metric(config, 2.0),
        ...     record_metric(config, 3.0),
        ... ]
        >>> result = aggregate_metrics(vals)
        >>> result["requests"]["_total"]
        6.0

        >>> config = create_metric_config("gauge", "temp", labels=["loc"])
        >>> vals = [
        ...     record_metric(config, 20.0, {"loc": "a"}),
        ...     record_metric(config, 25.0, {"loc": "a"}),
        ...     record_metric(config, 30.0, {"loc": "b"}),
        ... ]
        >>> result = aggregate_metrics(vals)
        >>> result["temp"]["loc=a"]
        25.0
        >>> result["temp"]["loc=b"]
        30.0

        >>> aggregate_metrics([])
        {}
    """
    if not values:
        return {}

    result: dict[str, dict[str, float]] = {}

    for val in values:
        name = val.config.name
        if name not in result:
            result[name] = {}

        # Create label key
        if val.labels:
            label_key = ",".join(f"{k}={v}" for k, v in sorted(val.labels.items()))
        else:
            label_key = "_total"

        # Aggregate based on metric type
        if val.config.metric_type == MetricType.COUNTER:
            # Sum for counters
            result[name][label_key] = result[name].get(label_key, 0.0) + val.value
        elif val.config.metric_type == MetricType.GAUGE:
            # Last value for gauges
            result[name][label_key] = val.value
        else:
            # For histogram/summary, just keep the last value for now
            result[name][label_key] = val.value

    return result


def export_telemetry(
    values: list[MetricValue],
    export_format: str = "json",
) -> str:
    """Export telemetry data in the specified format.

    Args:
        values: List of metric values to export.
        export_format: Export format. Defaults to "json".

    Returns:
        Formatted telemetry string.

    Raises:
        ValueError: If export_format is invalid.

    Examples:
        >>> config = create_metric_config("counter", "requests")
        >>> vals = [record_metric(config, 5.0)]
        >>> output = export_telemetry(vals, "json")
        >>> "requests" in output
        True

        >>> output = export_telemetry(vals, "prometheus")
        >>> "requests" in output
        True

        >>> export_telemetry([], "invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: export_format must be one of
    """
    if export_format not in VALID_EXPORT_FORMATS:
        fmt = export_format
        msg = f"export_format must be one of {VALID_EXPORT_FORMATS}, got '{fmt}'"
        raise ValueError(msg)

    aggregated = aggregate_metrics(values)

    if export_format == "json":
        return _export_json(values, aggregated)
    elif export_format == "prometheus":
        return _export_prometheus(values, aggregated)
    elif export_format == "otlp":
        return _export_otlp(values, aggregated)
    else:  # cloudwatch
        return _export_cloudwatch(values, aggregated)


def _export_json(
    values: list[MetricValue],
    aggregated: dict[str, dict[str, float]],
) -> str:
    """Export telemetry as JSON.

    Args:
        values: List of metric values.
        aggregated: Aggregated values.

    Returns:
        JSON string.

    Examples:
        >>> config = create_metric_config("counter", "test")
        >>> vals = [record_metric(config, 1.0)]
        >>> agg = aggregate_metrics(vals)
        >>> output = _export_json(vals, agg)
        >>> "test" in output
        True
    """
    data = {
        "metrics": aggregated,
        "timestamp": datetime.now(UTC).isoformat(),
        "count": len(values),
    }
    return json.dumps(data, indent=2)


def _export_prometheus(
    values: list[MetricValue],
    aggregated: dict[str, dict[str, float]],
) -> str:
    """Export telemetry in Prometheus text format.

    Args:
        values: List of metric values.
        aggregated: Aggregated values.

    Returns:
        Prometheus text format string.

    Examples:
        >>> config = create_metric_config("counter", "test_total")
        >>> vals = [record_metric(config, 1.0)]
        >>> agg = aggregate_metrics(vals)
        >>> output = _export_prometheus(vals, agg)
        >>> "test_total" in output
        True
    """
    lines = []

    for metric_name, label_values in aggregated.items():
        # Find the config for this metric
        config = None
        for val in values:
            if val.config.name == metric_name:
                config = val.config
                break

        if config:
            lines.append(f"# HELP {metric_name} {config.description}")
            lines.append(f"# TYPE {metric_name} {config.metric_type.value}")

        for label_key, value in label_values.items():
            if label_key == "_total":
                lines.append(f"{metric_name} {value}")
            else:
                # Parse label_key back to labels
                lbl = label_key.replace("=", '="').replace(",", '",')
                labels_str = "{" + lbl + '"}'
                lines.append(f"{metric_name}{labels_str} {value}")

    return "\n".join(lines)


def _export_otlp(
    values: list[MetricValue],
    aggregated: dict[str, dict[str, float]],
) -> str:
    """Export telemetry in OTLP-compatible JSON format.

    Args:
        values: List of metric values.
        aggregated: Aggregated values.

    Returns:
        OTLP JSON string.

    Examples:
        >>> config = create_metric_config("counter", "test")
        >>> vals = [record_metric(config, 1.0)]
        >>> agg = aggregate_metrics(vals)
        >>> output = _export_otlp(vals, agg)
        >>> "resourceMetrics" in output
        True
    """
    metrics_data = []

    for metric_name, label_values in aggregated.items():
        config = None
        for val in values:
            if val.config.name == metric_name:
                config = val.config
                break

        data_points = []
        for label_key, value in label_values.items():
            point = {
                "asDouble": value,
                "timeUnixNano": int(datetime.now(UTC).timestamp() * 1e9),
            }
            if label_key != "_total":
                # Parse labels
                attrs = []
                for kv in label_key.split(","):
                    k, v = kv.split("=")
                    attrs.append({"key": k, "value": {"stringValue": v}})
                point["attributes"] = attrs
            data_points.append(point)

        metric_type = config.metric_type.value if config else "gauge"
        metrics_data.append(
            {
                "name": metric_name,
                "description": config.description if config else "",
                metric_type: {"dataPoints": data_points},
            }
        )

    otlp = {
        "resourceMetrics": [
            {
                "scopeMetrics": [
                    {
                        "metrics": metrics_data,
                    }
                ],
            }
        ],
    }
    return json.dumps(otlp, indent=2)


def _export_cloudwatch(
    values: list[MetricValue],
    aggregated: dict[str, dict[str, float]],
) -> str:
    """Export telemetry in CloudWatch-compatible JSON format.

    Args:
        values: List of metric values.
        aggregated: Aggregated values.

    Returns:
        CloudWatch JSON string.

    Examples:
        >>> config = create_metric_config("counter", "test")
        >>> vals = [record_metric(config, 1.0)]
        >>> agg = aggregate_metrics(vals)
        >>> output = _export_cloudwatch(vals, agg)
        >>> "MetricData" in output
        True
    """
    metric_data = []
    timestamp = datetime.now(UTC).isoformat()

    for metric_name, label_values in aggregated.items():
        for label_key, value in label_values.items():
            datum = {
                "MetricName": metric_name,
                "Value": value,
                "Timestamp": timestamp,
                "Unit": "None",
            }

            if label_key != "_total":
                dimensions = []
                for kv in label_key.split(","):
                    k, v = kv.split("=")
                    dimensions.append({"Name": k, "Value": v})
                datum["Dimensions"] = dimensions

            metric_data.append(datum)

    cloudwatch = {
        "Namespace": "HuggingFace/Hub",
        "MetricData": metric_data,
    }
    return json.dumps(cloudwatch, indent=2)


def calculate_percentiles(
    values: list[float],
    percentiles: list[float] | None = None,
) -> dict[float, float]:
    """Calculate percentiles for a list of values.

    Args:
        values: List of numeric values.
        percentiles: Percentiles to calculate (0-100). Defaults to [50, 90, 95, 99].

    Returns:
        Dict mapping percentile to calculated value.

    Raises:
        ValueError: If values is empty or percentiles are invalid.

    Examples:
        >>> vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> result = calculate_percentiles(vals, [50, 90])
        >>> result[50]
        5.5
        >>> result[90]
        9.5

        >>> calculate_percentiles([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: values cannot be empty

        >>> calculate_percentiles([1], [101])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: percentile must be between 0 and 100
    """
    if not values:
        msg = "values cannot be empty"
        raise ValueError(msg)

    if percentiles is None:
        percentiles = [50.0, 90.0, 95.0, 99.0]

    for p in percentiles:
        if p < 0 or p > 100:
            msg = f"percentile must be between 0 and 100, got {p}"
            raise ValueError(msg)

    sorted_values = sorted(values)
    n = len(sorted_values)
    result: dict[float, float] = {}

    for p in percentiles:
        # Linear interpolation method
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower

        if lower == upper:
            result[p] = sorted_values[lower]
        else:
            result[p] = sorted_values[lower] + frac * (
                sorted_values[upper] - sorted_values[lower]
            )

    return result


def format_telemetry_stats(stats: TelemetryStats) -> str:
    """Format telemetry stats as a human-readable string.

    Args:
        stats: Telemetry stats to format.

    Returns:
        Formatted stats string.

    Examples:
        >>> stats = TelemetryStats(
        ...     total_events=1000,
        ...     events_by_level={"info": 800, "error": 200},
        ...     total_metrics=5000,
        ...     export_errors=2,
        ... )
        >>> output = format_telemetry_stats(stats)
        >>> "Total Events: 1000" in output
        True
        >>> "info: 800" in output
        True
    """
    lines = [
        "Telemetry Statistics",
        "=" * 40,
        f"Total Events: {stats.total_events}",
        f"Total Metrics: {stats.total_metrics}",
        f"Export Errors: {stats.export_errors}",
    ]

    if stats.events_by_level:
        lines.append("")
        lines.append("Events by Level:")
        for level in sorted(stats.events_by_level.keys()):
            count = stats.events_by_level[level]
            lines.append(f"  {level}: {count}")

    return "\n".join(lines)


def get_recommended_telemetry_config(
    use_case: str = "development",
) -> TelemetryConfig:
    """Get recommended telemetry configuration for a use case.

    Args:
        use_case: Use case (development, production, debugging).
            Defaults to "development".

    Returns:
        TelemetryConfig with recommended settings.

    Raises:
        ValueError: If use_case is unknown.

    Examples:
        >>> config = get_recommended_telemetry_config("development")
        >>> config.log_config.level
        <LogLevel.DEBUG: 'debug'>
        >>> config.batch_size
        10

        >>> config = get_recommended_telemetry_config("production")
        >>> config.log_config.level
        <LogLevel.INFO: 'info'>
        >>> config.batch_size
        100

        >>> config = get_recommended_telemetry_config("debugging")
        >>> config.log_config.include_context
        True

        >>> get_recommended_telemetry_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "invalid"
        ... )
        Traceback (most recent call last):
        ValueError: use_case must be one of
    """
    valid_use_cases = {"development", "production", "debugging"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    if use_case == "development":
        log_config = create_log_config(
            level="debug",
            include_timestamp=True,
            include_context=True,
        )
        metrics = [
            create_metric_config("counter", "requests_total", "Total requests"),
            create_metric_config(
                "histogram",
                "request_duration",
                "Request duration",
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            ),
        ]
        return create_telemetry_config(
            log_config=log_config,
            metrics=metrics,
            export_format="json",
            batch_size=10,
            flush_interval=5.0,
        )

    elif use_case == "production":
        log_config = create_log_config(
            level="info",
            include_timestamp=True,
            include_context=False,
        )
        metrics = [
            create_metric_config("counter", "requests_total", "Total requests"),
            create_metric_config("counter", "errors_total", "Total errors"),
            create_metric_config(
                "histogram",
                "request_duration",
                "Request duration",
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            ),
            create_metric_config("gauge", "active_connections", "Active connections"),
        ]
        return create_telemetry_config(
            log_config=log_config,
            metrics=metrics,
            export_format="prometheus",
            batch_size=100,
            flush_interval=30.0,
        )

    else:  # debugging
        log_config = create_log_config(
            level="debug",
            include_timestamp=True,
            include_context=True,
        )
        metrics = [
            create_metric_config("counter", "function_calls", "Function call count"),
            create_metric_config(
                "histogram",
                "function_duration",
                "Function execution time",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            ),
            create_metric_config("gauge", "memory_usage", "Memory usage in bytes"),
        ]
        return create_telemetry_config(
            log_config=log_config,
            metrics=metrics,
            export_format="json",
            batch_size=1,
            flush_interval=1.0,
        )
