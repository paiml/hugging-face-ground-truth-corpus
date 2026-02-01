"""Tests for inference.batching module."""

from __future__ import annotations

import pytest

from hf_gtc.inference.batching import (
    VALID_BATCHING_STRATEGIES,
    VALID_QUEUE_OVERFLOW_POLICIES,
    VALID_SCHEDULING_POLICIES,
    BatchConfig,
    BatchingStats,
    BatchingStrategy,
    LatencySLO,
    QueueConfig,
    QueueOverflowPolicy,
    SchedulingPolicy,
    calculate_optimal_batch_size,
    calculate_token_budget,
    check_slo_compliance,
    create_batch_config,
    create_batching_stats,
    create_latency_slo,
    create_queue_config,
    estimate_throughput,
    format_batching_stats,
    get_batching_strategy,
    get_queue_overflow_policy,
    get_recommended_batching_config,
    get_scheduling_policy,
    list_batching_strategies,
    list_queue_overflow_policies,
    list_scheduling_policies,
    validate_batch_config,
    validate_batching_stats,
    validate_latency_slo,
    validate_queue_config,
)


class TestBatchingStrategy:
    """Tests for BatchingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in BatchingStrategy:
            assert isinstance(strategy.value, str)

    def test_static_value(self) -> None:
        """STATIC has correct value."""
        assert BatchingStrategy.STATIC.value == "static"

    def test_dynamic_value(self) -> None:
        """DYNAMIC has correct value."""
        assert BatchingStrategy.DYNAMIC.value == "dynamic"

    def test_continuous_value(self) -> None:
        """CONTINUOUS has correct value."""
        assert BatchingStrategy.CONTINUOUS.value == "continuous"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_BATCHING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_BATCHING_STRATEGIES, frozenset)

    def test_valid_strategies_count(self) -> None:
        """VALID_BATCHING_STRATEGIES has correct count."""
        assert len(VALID_BATCHING_STRATEGIES) == 3


class TestSchedulingPolicy:
    """Tests for SchedulingPolicy enum."""

    def test_all_policies_have_values(self) -> None:
        """All policies have string values."""
        for policy in SchedulingPolicy:
            assert isinstance(policy.value, str)

    def test_fcfs_value(self) -> None:
        """FCFS has correct value."""
        assert SchedulingPolicy.FCFS.value == "fcfs"

    def test_sjf_value(self) -> None:
        """SJF has correct value."""
        assert SchedulingPolicy.SJF.value == "sjf"

    def test_priority_value(self) -> None:
        """PRIORITY has correct value."""
        assert SchedulingPolicy.PRIORITY.value == "priority"

    def test_fair_share_value(self) -> None:
        """FAIR_SHARE has correct value."""
        assert SchedulingPolicy.FAIR_SHARE.value == "fair_share"

    def test_valid_policies_frozenset(self) -> None:
        """VALID_SCHEDULING_POLICIES is a frozenset."""
        assert isinstance(VALID_SCHEDULING_POLICIES, frozenset)

    def test_valid_policies_count(self) -> None:
        """VALID_SCHEDULING_POLICIES has correct count."""
        assert len(VALID_SCHEDULING_POLICIES) == 4


class TestQueueOverflowPolicy:
    """Tests for QueueOverflowPolicy enum."""

    def test_all_policies_have_values(self) -> None:
        """All policies have string values."""
        for policy in QueueOverflowPolicy:
            assert isinstance(policy.value, str)

    def test_reject_value(self) -> None:
        """REJECT has correct value."""
        assert QueueOverflowPolicy.REJECT.value == "reject"

    def test_wait_value(self) -> None:
        """WAIT has correct value."""
        assert QueueOverflowPolicy.WAIT.value == "wait"

    def test_preempt_value(self) -> None:
        """PREEMPT has correct value."""
        assert QueueOverflowPolicy.PREEMPT.value == "preempt"

    def test_valid_policies_frozenset(self) -> None:
        """VALID_QUEUE_OVERFLOW_POLICIES is a frozenset."""
        assert isinstance(VALID_QUEUE_OVERFLOW_POLICIES, frozenset)

    def test_valid_policies_count(self) -> None:
        """VALID_QUEUE_OVERFLOW_POLICIES has correct count."""
        assert len(VALID_QUEUE_OVERFLOW_POLICIES) == 3


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_create_config(self) -> None:
        """Create batch config."""
        config = BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=32,
            max_tokens_per_batch=4096,
            padding_strategy="longest",
        )
        assert config.max_batch_size == 32
        assert config.strategy == BatchingStrategy.CONTINUOUS

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=32,
            max_tokens_per_batch=4096,
            padding_strategy="longest",
        )
        with pytest.raises(AttributeError):
            config.max_batch_size = 64  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        config = BatchConfig(
            strategy=BatchingStrategy.DYNAMIC,
            max_batch_size=16,
            max_tokens_per_batch=2048,
            padding_strategy="max_length",
        )
        assert config.strategy == BatchingStrategy.DYNAMIC
        assert config.max_batch_size == 16
        assert config.max_tokens_per_batch == 2048
        assert config.padding_strategy == "max_length"


class TestQueueConfig:
    """Tests for QueueConfig dataclass."""

    def test_create_config(self) -> None:
        """Create queue config."""
        config = QueueConfig(
            max_queue_size=1000,
            overflow_policy=QueueOverflowPolicy.REJECT,
            timeout_seconds=30.0,
        )
        assert config.max_queue_size == 1000
        assert config.overflow_policy == QueueOverflowPolicy.REJECT

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = QueueConfig(
            max_queue_size=1000,
            overflow_policy=QueueOverflowPolicy.REJECT,
            timeout_seconds=30.0,
        )
        with pytest.raises(AttributeError):
            config.max_queue_size = 2000  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        config = QueueConfig(
            max_queue_size=500,
            overflow_policy=QueueOverflowPolicy.WAIT,
            timeout_seconds=60.0,
        )
        assert config.max_queue_size == 500
        assert config.overflow_policy == QueueOverflowPolicy.WAIT
        assert config.timeout_seconds == pytest.approx(60.0)


class TestLatencySLO:
    """Tests for LatencySLO dataclass."""

    def test_create_slo(self) -> None:
        """Create latency SLO."""
        slo = LatencySLO(
            p50_ms=50.0,
            p90_ms=100.0,
            p99_ms=200.0,
            max_ms=500.0,
        )
        assert slo.p50_ms == pytest.approx(50.0)
        assert slo.p99_ms == pytest.approx(200.0)

    def test_slo_is_frozen(self) -> None:
        """SLO is immutable."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        with pytest.raises(AttributeError):
            slo.p50_ms = 25.0  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        slo = LatencySLO(25.0, 75.0, 150.0, 300.0)
        assert slo.p50_ms == pytest.approx(25.0)
        assert slo.p90_ms == pytest.approx(75.0)
        assert slo.p99_ms == pytest.approx(150.0)
        assert slo.max_ms == pytest.approx(300.0)


class TestBatchingStats:
    """Tests for BatchingStats dataclass."""

    def test_create_stats(self) -> None:
        """Create batching stats."""
        stats = BatchingStats(
            requests_processed=10000,
            avg_batch_size=24.5,
            avg_wait_time_ms=15.3,
            slo_violations=42,
        )
        assert stats.requests_processed == 10000
        assert stats.avg_batch_size == pytest.approx(24.5)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = BatchingStats(10000, 24.5, 15.3, 42)
        with pytest.raises(AttributeError):
            stats.requests_processed = 20000  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        stats = BatchingStats(5000, 16.0, 10.0, 5)
        assert stats.requests_processed == 5000
        assert stats.avg_batch_size == pytest.approx(16.0)
        assert stats.avg_wait_time_ms == pytest.approx(10.0)
        assert stats.slo_violations == 5


class TestValidateBatchConfig:
    """Tests for validate_batch_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=32,
            max_tokens_per_batch=4096,
            padding_strategy="longest",
        )
        validate_batch_config(config)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        config = BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=0,
            max_tokens_per_batch=4096,
            padding_strategy="longest",
        )
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            validate_batch_config(config)

    def test_negative_batch_size_raises(self) -> None:
        """Negative batch size raises ValueError."""
        config = BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=-1,
            max_tokens_per_batch=4096,
            padding_strategy="longest",
        )
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            validate_batch_config(config)

    def test_zero_tokens_raises(self) -> None:
        """Zero tokens per batch raises ValueError."""
        config = BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=32,
            max_tokens_per_batch=0,
            padding_strategy="longest",
        )
        with pytest.raises(ValueError, match="max_tokens_per_batch must be positive"):
            validate_batch_config(config)

    def test_invalid_padding_strategy_raises(self) -> None:
        """Invalid padding strategy raises ValueError."""
        config = BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=32,
            max_tokens_per_batch=4096,
            padding_strategy="invalid",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="padding_strategy must be one of"):
            validate_batch_config(config)


class TestValidateQueueConfig:
    """Tests for validate_queue_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = QueueConfig(
            max_queue_size=1000,
            overflow_policy=QueueOverflowPolicy.REJECT,
            timeout_seconds=30.0,
        )
        validate_queue_config(config)

    def test_zero_queue_size_raises(self) -> None:
        """Zero queue size raises ValueError."""
        config = QueueConfig(
            max_queue_size=0,
            overflow_policy=QueueOverflowPolicy.REJECT,
            timeout_seconds=30.0,
        )
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            validate_queue_config(config)

    def test_negative_queue_size_raises(self) -> None:
        """Negative queue size raises ValueError."""
        config = QueueConfig(
            max_queue_size=-1,
            overflow_policy=QueueOverflowPolicy.REJECT,
            timeout_seconds=30.0,
        )
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            validate_queue_config(config)

    def test_zero_timeout_raises(self) -> None:
        """Zero timeout raises ValueError."""
        config = QueueConfig(
            max_queue_size=1000,
            overflow_policy=QueueOverflowPolicy.REJECT,
            timeout_seconds=0.0,
        )
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_queue_config(config)

    def test_negative_timeout_raises(self) -> None:
        """Negative timeout raises ValueError."""
        config = QueueConfig(
            max_queue_size=1000,
            overflow_policy=QueueOverflowPolicy.REJECT,
            timeout_seconds=-1.0,
        )
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_queue_config(config)


class TestValidateLatencySLO:
    """Tests for validate_latency_slo function."""

    def test_valid_slo(self) -> None:
        """Valid SLO passes validation."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        validate_latency_slo(slo)

    def test_negative_p50_raises(self) -> None:
        """Negative p50 raises ValueError."""
        slo = LatencySLO(-1.0, 100.0, 200.0, 500.0)
        with pytest.raises(ValueError, match="p50_ms must be positive"):
            validate_latency_slo(slo)

    def test_zero_p50_raises(self) -> None:
        """Zero p50 raises ValueError."""
        slo = LatencySLO(0.0, 100.0, 200.0, 500.0)
        with pytest.raises(ValueError, match="p50_ms must be positive"):
            validate_latency_slo(slo)

    def test_negative_p90_raises(self) -> None:
        """Negative p90 raises ValueError."""
        slo = LatencySLO(50.0, -1.0, 200.0, 500.0)
        with pytest.raises(ValueError, match="p90_ms must be positive"):
            validate_latency_slo(slo)

    def test_negative_p99_raises(self) -> None:
        """Negative p99 raises ValueError."""
        slo = LatencySLO(50.0, 100.0, -1.0, 500.0)
        with pytest.raises(ValueError, match="p99_ms must be positive"):
            validate_latency_slo(slo)

    def test_negative_max_raises(self) -> None:
        """Negative max raises ValueError."""
        slo = LatencySLO(50.0, 100.0, 200.0, -1.0)
        with pytest.raises(ValueError, match="max_ms must be positive"):
            validate_latency_slo(slo)

    def test_p90_less_than_p50_raises(self) -> None:
        """p90 < p50 raises ValueError."""
        slo = LatencySLO(100.0, 50.0, 200.0, 500.0)
        with pytest.raises(ValueError, match="p90_ms must be >= p50_ms"):
            validate_latency_slo(slo)

    def test_p99_less_than_p90_raises(self) -> None:
        """p99 < p90 raises ValueError."""
        slo = LatencySLO(50.0, 200.0, 100.0, 500.0)
        with pytest.raises(ValueError, match="p99_ms must be >= p90_ms"):
            validate_latency_slo(slo)

    def test_max_less_than_p99_raises(self) -> None:
        """Max < p99 raises ValueError."""
        slo = LatencySLO(50.0, 100.0, 500.0, 200.0)
        with pytest.raises(ValueError, match="max_ms must be >= p99_ms"):
            validate_latency_slo(slo)

    def test_equal_values_valid(self) -> None:
        """Equal values are valid."""
        slo = LatencySLO(100.0, 100.0, 100.0, 100.0)
        validate_latency_slo(slo)


class TestValidateBatchingStats:
    """Tests for validate_batching_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = BatchingStats(10000, 24.5, 15.3, 42)
        validate_batching_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        stats = BatchingStats(0, 0.0, 0.0, 0)
        validate_batching_stats(stats)

    def test_negative_requests_raises(self) -> None:
        """Negative requests raises ValueError."""
        stats = BatchingStats(-1, 24.5, 15.3, 42)
        with pytest.raises(ValueError, match="requests_processed cannot be negative"):
            validate_batching_stats(stats)

    def test_negative_avg_batch_size_raises(self) -> None:
        """Negative avg batch size raises ValueError."""
        stats = BatchingStats(10000, -1.0, 15.3, 42)
        with pytest.raises(ValueError, match="avg_batch_size cannot be negative"):
            validate_batching_stats(stats)

    def test_negative_avg_wait_time_raises(self) -> None:
        """Negative avg wait time raises ValueError."""
        stats = BatchingStats(10000, 24.5, -1.0, 42)
        with pytest.raises(ValueError, match="avg_wait_time_ms cannot be negative"):
            validate_batching_stats(stats)

    def test_negative_slo_violations_raises(self) -> None:
        """Negative SLO violations raises ValueError."""
        stats = BatchingStats(10000, 24.5, 15.3, -1)
        with pytest.raises(ValueError, match="slo_violations cannot be negative"):
            validate_batching_stats(stats)


class TestCreateBatchConfig:
    """Tests for create_batch_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_batch_config()
        assert config.strategy == BatchingStrategy.CONTINUOUS
        assert config.max_batch_size == 32
        assert config.max_tokens_per_batch == 4096
        assert config.padding_strategy == "longest"

    def test_custom_strategy(self) -> None:
        """Create config with custom strategy."""
        config = create_batch_config(strategy="static")
        assert config.strategy == BatchingStrategy.STATIC

    def test_custom_batch_size(self) -> None:
        """Create config with custom batch size."""
        config = create_batch_config(max_batch_size=64)
        assert config.max_batch_size == 64

    def test_custom_tokens(self) -> None:
        """Create config with custom tokens per batch."""
        config = create_batch_config(max_tokens_per_batch=8192)
        assert config.max_tokens_per_batch == 8192

    def test_custom_padding(self) -> None:
        """Create config with custom padding strategy."""
        config = create_batch_config(padding_strategy="max_length")
        assert config.padding_strategy == "max_length"

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_batch_config(strategy="invalid")  # type: ignore[arg-type]

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            create_batch_config(max_batch_size=0)


class TestCreateQueueConfig:
    """Tests for create_queue_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_queue_config()
        assert config.max_queue_size == 1000
        assert config.overflow_policy == QueueOverflowPolicy.REJECT
        assert config.timeout_seconds == pytest.approx(30.0)

    def test_custom_queue_size(self) -> None:
        """Create config with custom queue size."""
        config = create_queue_config(max_queue_size=500)
        assert config.max_queue_size == 500

    def test_custom_overflow_policy(self) -> None:
        """Create config with custom overflow policy."""
        config = create_queue_config(overflow_policy="wait")
        assert config.overflow_policy == QueueOverflowPolicy.WAIT

    def test_custom_timeout(self) -> None:
        """Create config with custom timeout."""
        config = create_queue_config(timeout_seconds=60.0)
        assert config.timeout_seconds == pytest.approx(60.0)

    def test_invalid_overflow_policy_raises(self) -> None:
        """Invalid overflow policy raises ValueError."""
        with pytest.raises(ValueError, match="overflow_policy must be one of"):
            create_queue_config(overflow_policy="invalid")  # type: ignore[arg-type]

    def test_zero_queue_size_raises(self) -> None:
        """Zero queue size raises ValueError."""
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            create_queue_config(max_queue_size=0)


class TestCreateLatencySLO:
    """Tests for create_latency_slo function."""

    def test_default_slo(self) -> None:
        """Create default SLO."""
        slo = create_latency_slo()
        assert slo.p50_ms == pytest.approx(50.0)
        assert slo.p90_ms == pytest.approx(100.0)
        assert slo.p99_ms == pytest.approx(200.0)
        assert slo.max_ms == pytest.approx(500.0)

    def test_custom_p50(self) -> None:
        """Create SLO with custom p50."""
        slo = create_latency_slo(p50_ms=25.0)
        assert slo.p50_ms == pytest.approx(25.0)

    def test_custom_p90(self) -> None:
        """Create SLO with custom p90."""
        slo = create_latency_slo(p90_ms=150.0)
        assert slo.p90_ms == pytest.approx(150.0)

    def test_custom_p99(self) -> None:
        """Create SLO with custom p99."""
        slo = create_latency_slo(p99_ms=150.0)
        assert slo.p99_ms == pytest.approx(150.0)

    def test_custom_max(self) -> None:
        """Create SLO with custom max."""
        slo = create_latency_slo(max_ms=1000.0)
        assert slo.max_ms == pytest.approx(1000.0)

    def test_negative_p50_raises(self) -> None:
        """Negative p50 raises ValueError."""
        with pytest.raises(ValueError, match="p50_ms must be positive"):
            create_latency_slo(p50_ms=-1.0)


class TestCreateBatchingStats:
    """Tests for create_batching_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_batching_stats()
        assert stats.requests_processed == 0
        assert stats.avg_batch_size == pytest.approx(0.0)
        assert stats.avg_wait_time_ms == pytest.approx(0.0)
        assert stats.slo_violations == 0

    def test_custom_requests(self) -> None:
        """Create stats with custom requests."""
        stats = create_batching_stats(requests_processed=1000)
        assert stats.requests_processed == 1000

    def test_custom_batch_size(self) -> None:
        """Create stats with custom avg batch size."""
        stats = create_batching_stats(avg_batch_size=28.5)
        assert stats.avg_batch_size == pytest.approx(28.5)

    def test_custom_wait_time(self) -> None:
        """Create stats with custom avg wait time."""
        stats = create_batching_stats(avg_wait_time_ms=10.0)
        assert stats.avg_wait_time_ms == pytest.approx(10.0)

    def test_custom_violations(self) -> None:
        """Create stats with custom violations."""
        stats = create_batching_stats(slo_violations=5)
        assert stats.slo_violations == 5

    def test_negative_requests_raises(self) -> None:
        """Negative requests raises ValueError."""
        with pytest.raises(ValueError, match="requests_processed cannot be negative"):
            create_batching_stats(requests_processed=-1)


class TestListBatchingStrategies:
    """Tests for list_batching_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_batching_strategies()
        assert strategies == sorted(strategies)

    def test_contains_continuous(self) -> None:
        """Contains continuous."""
        strategies = list_batching_strategies()
        assert "continuous" in strategies

    def test_contains_static(self) -> None:
        """Contains static."""
        strategies = list_batching_strategies()
        assert "static" in strategies

    def test_contains_all_strategies(self) -> None:
        """Contains all strategies."""
        strategies = list_batching_strategies()
        assert len(strategies) == len(VALID_BATCHING_STRATEGIES)


class TestListSchedulingPolicies:
    """Tests for list_scheduling_policies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        policies = list_scheduling_policies()
        assert policies == sorted(policies)

    def test_contains_fcfs(self) -> None:
        """Contains fcfs."""
        policies = list_scheduling_policies()
        assert "fcfs" in policies

    def test_contains_fair_share(self) -> None:
        """Contains fair_share."""
        policies = list_scheduling_policies()
        assert "fair_share" in policies

    def test_contains_all_policies(self) -> None:
        """Contains all policies."""
        policies = list_scheduling_policies()
        assert len(policies) == len(VALID_SCHEDULING_POLICIES)


class TestListQueueOverflowPolicies:
    """Tests for list_queue_overflow_policies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        policies = list_queue_overflow_policies()
        assert policies == sorted(policies)

    def test_contains_reject(self) -> None:
        """Contains reject."""
        policies = list_queue_overflow_policies()
        assert "reject" in policies

    def test_contains_preempt(self) -> None:
        """Contains preempt."""
        policies = list_queue_overflow_policies()
        assert "preempt" in policies

    def test_contains_all_policies(self) -> None:
        """Contains all policies."""
        policies = list_queue_overflow_policies()
        assert len(policies) == len(VALID_QUEUE_OVERFLOW_POLICIES)


class TestGetBatchingStrategy:
    """Tests for get_batching_strategy function."""

    def test_get_continuous(self) -> None:
        """Get continuous strategy."""
        strategy = get_batching_strategy("continuous")
        assert strategy == BatchingStrategy.CONTINUOUS

    def test_get_static(self) -> None:
        """Get static strategy."""
        strategy = get_batching_strategy("static")
        assert strategy == BatchingStrategy.STATIC

    def test_get_dynamic(self) -> None:
        """Get dynamic strategy."""
        strategy = get_batching_strategy("dynamic")
        assert strategy == BatchingStrategy.DYNAMIC

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown batching strategy"):
            get_batching_strategy("invalid")


class TestGetSchedulingPolicy:
    """Tests for get_scheduling_policy function."""

    def test_get_fcfs(self) -> None:
        """Get fcfs policy."""
        policy = get_scheduling_policy("fcfs")
        assert policy == SchedulingPolicy.FCFS

    def test_get_sjf(self) -> None:
        """Get sjf policy."""
        policy = get_scheduling_policy("sjf")
        assert policy == SchedulingPolicy.SJF

    def test_get_priority(self) -> None:
        """Get priority policy."""
        policy = get_scheduling_policy("priority")
        assert policy == SchedulingPolicy.PRIORITY

    def test_get_fair_share(self) -> None:
        """Get fair_share policy."""
        policy = get_scheduling_policy("fair_share")
        assert policy == SchedulingPolicy.FAIR_SHARE

    def test_invalid_policy_raises(self) -> None:
        """Invalid policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduling policy"):
            get_scheduling_policy("invalid")


class TestGetQueueOverflowPolicy:
    """Tests for get_queue_overflow_policy function."""

    def test_get_reject(self) -> None:
        """Get reject policy."""
        policy = get_queue_overflow_policy("reject")
        assert policy == QueueOverflowPolicy.REJECT

    def test_get_wait(self) -> None:
        """Get wait policy."""
        policy = get_queue_overflow_policy("wait")
        assert policy == QueueOverflowPolicy.WAIT

    def test_get_preempt(self) -> None:
        """Get preempt policy."""
        policy = get_queue_overflow_policy("preempt")
        assert policy == QueueOverflowPolicy.PREEMPT

    def test_invalid_policy_raises(self) -> None:
        """Invalid policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown queue overflow policy"):
            get_queue_overflow_policy("invalid")


class TestCalculateOptimalBatchSize:
    """Tests for calculate_optimal_batch_size function."""

    def test_basic_calculation(self) -> None:
        """Basic batch size calculation."""
        batch_size = calculate_optimal_batch_size(
            available_memory_mb=1000.0,
            avg_sequence_length=512,
            hidden_size=768,
        )
        assert batch_size > 0

    def test_more_memory_larger_batch(self) -> None:
        """More memory allows larger batch."""
        size1 = calculate_optimal_batch_size(500.0, 512)
        size2 = calculate_optimal_batch_size(1000.0, 512)
        assert size2 > size1

    def test_longer_sequences_smaller_batch(self) -> None:
        """Longer sequences reduce batch size."""
        size512 = calculate_optimal_batch_size(1000.0, 512)
        size1024 = calculate_optimal_batch_size(1000.0, 1024)
        assert size512 > size1024

    def test_minimum_batch_size_one(self) -> None:
        """Minimum batch size is 1."""
        size = calculate_optimal_batch_size(0.001, 512)
        assert size >= 1

    def test_zero_memory_raises(self) -> None:
        """Zero memory raises ValueError."""
        with pytest.raises(ValueError, match="available_memory_mb must be positive"):
            calculate_optimal_batch_size(0.0, 512)

    def test_negative_memory_raises(self) -> None:
        """Negative memory raises ValueError."""
        with pytest.raises(ValueError, match="available_memory_mb must be positive"):
            calculate_optimal_batch_size(-1.0, 512)

    def test_zero_sequence_length_raises(self) -> None:
        """Zero sequence length raises ValueError."""
        with pytest.raises(ValueError, match="avg_sequence_length must be positive"):
            calculate_optimal_batch_size(1000.0, 0)

    def test_zero_hidden_size_raises(self) -> None:
        """Zero hidden size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            calculate_optimal_batch_size(1000.0, 512, hidden_size=0)

    def test_zero_dtype_bytes_raises(self) -> None:
        """Zero dtype bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            calculate_optimal_batch_size(1000.0, 512, dtype_bytes=0)

    def test_invalid_memory_fraction_raises(self) -> None:
        """Invalid memory fraction raises ValueError."""
        with pytest.raises(ValueError, match="memory_fraction must be in"):
            calculate_optimal_batch_size(1000.0, 512, memory_fraction=1.5)

    def test_custom_memory_fraction(self) -> None:
        """Custom memory fraction works."""
        size_low = calculate_optimal_batch_size(1000.0, 512, memory_fraction=0.5)
        size_high = calculate_optimal_batch_size(1000.0, 512, memory_fraction=1.0)
        assert size_high > size_low


class TestEstimateThroughput:
    """Tests for estimate_throughput function."""

    def test_basic_calculation(self) -> None:
        """Basic throughput calculation."""
        throughput = estimate_throughput(
            batch_size=32,
            avg_sequence_length=512,
            inference_time_ms=100.0,
        )
        assert throughput > 0

    def test_expected_value(self) -> None:
        """Expected throughput value."""
        # 32 * 512 = 16384 tokens per batch
        # 1000 / 100 = 10 batches per second
        # 16384 * 10 = 163840 tokens per second
        throughput = estimate_throughput(32, 512, 100.0)
        assert throughput == pytest.approx(163840.0)

    def test_larger_batch_higher_throughput(self) -> None:
        """Larger batch gives higher throughput."""
        throughput32 = estimate_throughput(32, 512, 100.0)
        throughput64 = estimate_throughput(64, 512, 100.0)
        assert throughput64 > throughput32

    def test_faster_inference_higher_throughput(self) -> None:
        """Faster inference gives higher throughput."""
        throughput100 = estimate_throughput(32, 512, 100.0)
        throughput50 = estimate_throughput(32, 512, 50.0)
        assert throughput50 > throughput100

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_throughput(0, 512, 100.0)

    def test_zero_sequence_length_raises(self) -> None:
        """Zero sequence length raises ValueError."""
        with pytest.raises(ValueError, match="avg_sequence_length must be positive"):
            estimate_throughput(32, 0, 100.0)

    def test_zero_inference_time_raises(self) -> None:
        """Zero inference time raises ValueError."""
        with pytest.raises(ValueError, match="inference_time_ms must be positive"):
            estimate_throughput(32, 512, 0.0)

    def test_negative_inference_time_raises(self) -> None:
        """Negative inference time raises ValueError."""
        with pytest.raises(ValueError, match="inference_time_ms must be positive"):
            estimate_throughput(32, 512, -1.0)


class TestCalculateTokenBudget:
    """Tests for calculate_token_budget function."""

    def test_basic_calculation(self) -> None:
        """Basic token budget calculation."""
        budget = calculate_token_budget(32, 512)
        assert budget > 0

    def test_expected_value(self) -> None:
        """Expected token budget value."""
        # 32 * 512 * 0.9 = 14745.6 -> 14745
        budget = calculate_token_budget(32, 512)
        assert budget == 14745

    def test_full_budget(self) -> None:
        """Full budget (fraction=1.0)."""
        budget = calculate_token_budget(32, 512, budget_fraction=1.0)
        assert budget == 16384

    def test_larger_batch_larger_budget(self) -> None:
        """Larger batch gives larger budget."""
        budget32 = calculate_token_budget(32, 512)
        budget64 = calculate_token_budget(64, 512)
        assert budget64 > budget32

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            calculate_token_budget(0, 512)

    def test_zero_sequence_length_raises(self) -> None:
        """Zero sequence length raises ValueError."""
        with pytest.raises(ValueError, match="max_sequence_length must be positive"):
            calculate_token_budget(32, 0)

    def test_invalid_budget_fraction_raises(self) -> None:
        """Invalid budget fraction raises ValueError."""
        with pytest.raises(ValueError, match="budget_fraction must be in"):
            calculate_token_budget(32, 512, budget_fraction=1.5)

    def test_zero_budget_fraction_raises(self) -> None:
        """Zero budget fraction raises ValueError."""
        with pytest.raises(ValueError, match="budget_fraction must be in"):
            calculate_token_budget(32, 512, budget_fraction=0.0)


class TestCheckSLOCompliance:
    """Tests for check_slo_compliance function."""

    def test_p99_compliant(self) -> None:
        """P99 compliant latency returns True."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(150.0, slo, "p99") is True

    def test_p99_non_compliant(self) -> None:
        """P99 non-compliant latency returns False."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(250.0, slo, "p99") is False

    def test_p50_compliant(self) -> None:
        """P50 compliant latency returns True."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(40.0, slo, "p50") is True

    def test_p50_non_compliant(self) -> None:
        """P50 non-compliant latency returns False."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(75.0, slo, "p50") is False

    def test_p90_compliant(self) -> None:
        """P90 compliant latency returns True."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(90.0, slo, "p90") is True

    def test_max_compliant(self) -> None:
        """Max compliant latency returns True."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(450.0, slo, "max") is True

    def test_max_non_compliant(self) -> None:
        """Max non-compliant latency returns False."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(600.0, slo, "max") is False

    def test_exact_threshold_compliant(self) -> None:
        """Exact threshold is compliant."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(200.0, slo, "p99") is True

    def test_zero_latency_compliant(self) -> None:
        """Zero latency is compliant."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        assert check_slo_compliance(0.0, slo, "p50") is True

    def test_negative_latency_raises(self) -> None:
        """Negative latency raises ValueError."""
        slo = LatencySLO(50.0, 100.0, 200.0, 500.0)
        with pytest.raises(ValueError, match="actual_latency_ms cannot be negative"):
            check_slo_compliance(-1.0, slo)


class TestFormatBatchingStats:
    """Tests for format_batching_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = BatchingStats(
            requests_processed=10000,
            avg_batch_size=24.5,
            avg_wait_time_ms=15.3,
            slo_violations=42,
        )
        formatted = format_batching_stats(stats)
        assert "Requests Processed: 10000" in formatted
        assert "Avg Batch Size: 24.50" in formatted
        assert "Avg Wait Time: 15.30 ms" in formatted
        assert "SLO Violations: 42" in formatted

    def test_zero_values(self) -> None:
        """Zero values format correctly."""
        stats = BatchingStats(0, 0.0, 0.0, 0)
        formatted = format_batching_stats(stats)
        assert "Requests Processed: 0" in formatted
        assert "SLO Violations: 0" in formatted

    def test_multiline_output(self) -> None:
        """Output contains multiple lines."""
        stats = BatchingStats(10000, 24.5, 15.3, 42)
        formatted = format_batching_stats(stats)
        lines = formatted.split("\n")
        assert len(lines) == 4


class TestGetRecommendedBatchingConfig:
    """Tests for get_recommended_batching_config function."""

    def test_cpu_config(self) -> None:
        """CPU config uses static batching."""
        config = get_recommended_batching_config("cpu")
        assert config.strategy == BatchingStrategy.STATIC
        assert config.max_batch_size == 8

    def test_gpu_consumer_config(self) -> None:
        """GPU consumer config uses dynamic batching."""
        config = get_recommended_batching_config("gpu_consumer")
        assert config.strategy == BatchingStrategy.DYNAMIC
        assert config.max_batch_size == 16

    def test_gpu_datacenter_config(self) -> None:
        """GPU datacenter config uses continuous batching."""
        config = get_recommended_batching_config("gpu_datacenter")
        assert config.strategy == BatchingStrategy.CONTINUOUS
        assert config.max_batch_size == 64

    def test_tpu_config(self) -> None:
        """TPU config uses continuous batching with max_length padding."""
        config = get_recommended_batching_config("tpu")
        assert config.strategy == BatchingStrategy.CONTINUOUS
        assert config.max_batch_size == 128
        assert config.padding_strategy == "max_length"

    def test_default_is_datacenter(self) -> None:
        """Default is gpu_datacenter."""
        config = get_recommended_batching_config()
        assert config.strategy == BatchingStrategy.CONTINUOUS
        assert config.max_batch_size == 64

    def test_invalid_hardware_raises(self) -> None:
        """Invalid hardware raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hardware type"):
            get_recommended_batching_config("invalid")  # type: ignore[arg-type]


class TestAllBatchingStrategies:
    """Test all batching strategies can be created."""

    @pytest.mark.parametrize("strategy", list(VALID_BATCHING_STRATEGIES))
    def test_create_config_with_strategy(self, strategy: str) -> None:
        """Config can be created with each strategy."""
        config = create_batch_config(strategy=strategy)  # type: ignore[arg-type]
        assert config.strategy.value == strategy


class TestAllQueueOverflowPolicies:
    """Test all queue overflow policies can be created."""

    @pytest.mark.parametrize("policy", list(VALID_QUEUE_OVERFLOW_POLICIES))
    def test_create_config_with_policy(self, policy: str) -> None:
        """Config can be created with each policy."""
        config = create_queue_config(overflow_policy=policy)  # type: ignore[arg-type]
        assert config.overflow_policy.value == policy


class TestAllSchedulingPolicies:
    """Test all scheduling policies can be retrieved."""

    @pytest.mark.parametrize("policy", list(VALID_SCHEDULING_POLICIES))
    def test_get_policy(self, policy: str) -> None:
        """Policy can be retrieved."""
        result = get_scheduling_policy(policy)
        assert result.value == policy
