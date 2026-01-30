"""Tests for data pipeline orchestration functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.pipeline import (
    VALID_CACHE_STRATEGIES,
    VALID_EXECUTION_MODES,
    VALID_PIPELINE_STAGES,
    CacheStrategy,
    ExecutionMode,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    PipelineStats,
    StageConfig,
    build_pipeline,
    create_pipeline_config,
    create_pipeline_result,
    create_pipeline_stats,
    create_stage_config,
    estimate_pipeline_memory,
    execute_pipeline,
    format_pipeline_stats,
    get_cache_strategy,
    get_execution_mode,
    get_pipeline_stage,
    get_recommended_pipeline_config,
    list_cache_strategies,
    list_execution_modes,
    list_pipeline_stages,
    optimize_pipeline,
    validate_pipeline_config,
    validate_pipeline_output,
    validate_pipeline_result,
    validate_pipeline_stats,
    validate_stage_config,
)


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_load_value(self) -> None:
        """Test LOAD stage value."""
        assert PipelineStage.LOAD.value == "load"

    def test_transform_value(self) -> None:
        """Test TRANSFORM stage value."""
        assert PipelineStage.TRANSFORM.value == "transform"

    def test_filter_value(self) -> None:
        """Test FILTER stage value."""
        assert PipelineStage.FILTER.value == "filter"

    def test_map_value(self) -> None:
        """Test MAP stage value."""
        assert PipelineStage.MAP.value == "map"

    def test_batch_value(self) -> None:
        """Test BATCH stage value."""
        assert PipelineStage.BATCH.value == "batch"

    def test_cache_value(self) -> None:
        """Test CACHE stage value."""
        assert PipelineStage.CACHE.value == "cache"

    def test_valid_stages_frozenset(self) -> None:
        """Test VALID_PIPELINE_STAGES contains all stages."""
        assert "load" in VALID_PIPELINE_STAGES
        assert "transform" in VALID_PIPELINE_STAGES
        assert "filter" in VALID_PIPELINE_STAGES
        assert "map" in VALID_PIPELINE_STAGES
        assert "batch" in VALID_PIPELINE_STAGES
        assert "cache" in VALID_PIPELINE_STAGES


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_eager_value(self) -> None:
        """Test EAGER mode value."""
        assert ExecutionMode.EAGER.value == "eager"

    def test_lazy_value(self) -> None:
        """Test LAZY mode value."""
        assert ExecutionMode.LAZY.value == "lazy"

    def test_parallel_value(self) -> None:
        """Test PARALLEL mode value."""
        assert ExecutionMode.PARALLEL.value == "parallel"

    def test_distributed_value(self) -> None:
        """Test DISTRIBUTED mode value."""
        assert ExecutionMode.DISTRIBUTED.value == "distributed"

    def test_valid_modes_frozenset(self) -> None:
        """Test VALID_EXECUTION_MODES contains all modes."""
        assert "eager" in VALID_EXECUTION_MODES
        assert "lazy" in VALID_EXECUTION_MODES
        assert "parallel" in VALID_EXECUTION_MODES
        assert "distributed" in VALID_EXECUTION_MODES


class TestCacheStrategy:
    """Tests for CacheStrategy enum."""

    def test_none_value(self) -> None:
        """Test NONE strategy value."""
        assert CacheStrategy.NONE.value == "none"

    def test_memory_value(self) -> None:
        """Test MEMORY strategy value."""
        assert CacheStrategy.MEMORY.value == "memory"

    def test_disk_value(self) -> None:
        """Test DISK strategy value."""
        assert CacheStrategy.DISK.value == "disk"

    def test_hybrid_value(self) -> None:
        """Test HYBRID strategy value."""
        assert CacheStrategy.HYBRID.value == "hybrid"

    def test_valid_strategies_frozenset(self) -> None:
        """Test VALID_CACHE_STRATEGIES contains all strategies."""
        assert "none" in VALID_CACHE_STRATEGIES
        assert "memory" in VALID_CACHE_STRATEGIES
        assert "disk" in VALID_CACHE_STRATEGIES
        assert "hybrid" in VALID_CACHE_STRATEGIES


class TestStageConfig:
    """Tests for StageConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating StageConfig instance."""
        config = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="tokenize",
            params={"max_length": 512},
            cache=CacheStrategy.MEMORY,
        )
        assert config.stage_type == PipelineStage.MAP
        assert config.function_name == "tokenize"
        assert config.params == {"max_length": 512}
        assert config.cache == CacheStrategy.MEMORY

    def test_frozen(self) -> None:
        """Test that StageConfig is immutable."""
        config = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="process",
            params={},
            cache=CacheStrategy.NONE,
        )
        with pytest.raises(AttributeError):
            config.function_name = "other"  # type: ignore[misc]


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PipelineConfig instance."""
        stage = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="process",
            params={},
            cache=CacheStrategy.NONE,
        )
        config = PipelineConfig(
            stages=(stage,),
            execution_mode=ExecutionMode.EAGER,
            num_workers=4,
            prefetch_factor=2,
        )
        assert len(config.stages) == 1
        assert config.execution_mode == ExecutionMode.EAGER
        assert config.num_workers == 4
        assert config.prefetch_factor == 2

    def test_frozen(self) -> None:
        """Test that PipelineConfig is immutable."""
        stage = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="process",
            params={},
            cache=CacheStrategy.NONE,
        )
        config = PipelineConfig(
            stages=(stage,),
            execution_mode=ExecutionMode.EAGER,
            num_workers=4,
            prefetch_factor=2,
        )
        with pytest.raises(AttributeError):
            config.num_workers = 8  # type: ignore[misc]


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_creation(self) -> None:
        """Test creating PipelineResult instance."""
        result = PipelineResult(
            num_processed=10000,
            num_filtered=500,
            processing_time_seconds=5.5,
            cache_hits=1000,
        )
        assert result.num_processed == 10000
        assert result.num_filtered == 500
        assert result.processing_time_seconds == pytest.approx(5.5)
        assert result.cache_hits == 1000

    def test_frozen(self) -> None:
        """Test that PipelineResult is immutable."""
        result = PipelineResult(
            num_processed=100,
            num_filtered=10,
            processing_time_seconds=1.0,
            cache_hits=50,
        )
        with pytest.raises(AttributeError):
            result.num_processed = 200  # type: ignore[misc]


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_creation(self) -> None:
        """Test creating PipelineStats instance."""
        stats = PipelineStats(
            total_samples=50000,
            throughput_samples_per_sec=10000.0,
            memory_usage_mb=256.0,
            cache_hit_rate=0.85,
        )
        assert stats.total_samples == 50000
        assert stats.throughput_samples_per_sec == pytest.approx(10000.0)
        assert stats.memory_usage_mb == pytest.approx(256.0)
        assert stats.cache_hit_rate == pytest.approx(0.85)

    def test_frozen(self) -> None:
        """Test that PipelineStats is immutable."""
        stats = PipelineStats(
            total_samples=1000,
            throughput_samples_per_sec=500.0,
            memory_usage_mb=128.0,
            cache_hit_rate=0.5,
        )
        with pytest.raises(AttributeError):
            stats.total_samples = 2000  # type: ignore[misc]


class TestValidateStageConfig:
    """Tests for validate_stage_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="process",
            params={},
            cache=CacheStrategy.NONE,
        )
        validate_stage_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_stage_config(None)  # type: ignore[arg-type]

    def test_empty_function_name_raises_error(self) -> None:
        """Test that empty function_name raises ValueError."""
        config = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="",
            params={},
            cache=CacheStrategy.NONE,
        )
        with pytest.raises(ValueError, match="function_name cannot be empty"):
            validate_stage_config(config)


class TestValidatePipelineConfig:
    """Tests for validate_pipeline_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        stage = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="process",
            params={},
            cache=CacheStrategy.NONE,
        )
        config = PipelineConfig(
            stages=(stage,),
            execution_mode=ExecutionMode.EAGER,
            num_workers=4,
            prefetch_factor=2,
        )
        validate_pipeline_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_pipeline_config(None)  # type: ignore[arg-type]

    def test_empty_stages_raises_error(self) -> None:
        """Test that empty stages raises ValueError."""
        config = PipelineConfig(
            stages=(),
            execution_mode=ExecutionMode.EAGER,
            num_workers=4,
            prefetch_factor=2,
        )
        with pytest.raises(ValueError, match="stages cannot be empty"):
            validate_pipeline_config(config)

    def test_zero_workers_raises_error(self) -> None:
        """Test that zero num_workers raises ValueError."""
        stage = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="process",
            params={},
            cache=CacheStrategy.NONE,
        )
        config = PipelineConfig(
            stages=(stage,),
            execution_mode=ExecutionMode.EAGER,
            num_workers=0,
            prefetch_factor=2,
        )
        with pytest.raises(ValueError, match="num_workers must be positive"):
            validate_pipeline_config(config)

    def test_negative_prefetch_raises_error(self) -> None:
        """Test that negative prefetch_factor raises ValueError."""
        stage = StageConfig(
            stage_type=PipelineStage.MAP,
            function_name="process",
            params={},
            cache=CacheStrategy.NONE,
        )
        config = PipelineConfig(
            stages=(stage,),
            execution_mode=ExecutionMode.EAGER,
            num_workers=4,
            prefetch_factor=-1,
        )
        with pytest.raises(ValueError, match="prefetch_factor cannot be negative"):
            validate_pipeline_config(config)


class TestValidatePipelineResult:
    """Tests for validate_pipeline_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = PipelineResult(
            num_processed=100,
            num_filtered=10,
            processing_time_seconds=1.0,
            cache_hits=50,
        )
        validate_pipeline_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_pipeline_result(None)  # type: ignore[arg-type]

    def test_negative_num_processed_raises_error(self) -> None:
        """Test that negative num_processed raises ValueError."""
        result = PipelineResult(
            num_processed=-1,
            num_filtered=0,
            processing_time_seconds=1.0,
            cache_hits=0,
        )
        with pytest.raises(ValueError, match="num_processed cannot be negative"):
            validate_pipeline_result(result)

    def test_negative_num_filtered_raises_error(self) -> None:
        """Test that negative num_filtered raises ValueError."""
        result = PipelineResult(
            num_processed=100,
            num_filtered=-1,
            processing_time_seconds=1.0,
            cache_hits=0,
        )
        with pytest.raises(ValueError, match="num_filtered cannot be negative"):
            validate_pipeline_result(result)

    def test_negative_processing_time_raises_error(self) -> None:
        """Test that negative processing_time_seconds raises ValueError."""
        result = PipelineResult(
            num_processed=100,
            num_filtered=0,
            processing_time_seconds=-1.0,
            cache_hits=0,
        )
        with pytest.raises(
            ValueError, match="processing_time_seconds cannot be negative"
        ):
            validate_pipeline_result(result)

    def test_negative_cache_hits_raises_error(self) -> None:
        """Test that negative cache_hits raises ValueError."""
        result = PipelineResult(
            num_processed=100,
            num_filtered=0,
            processing_time_seconds=1.0,
            cache_hits=-1,
        )
        with pytest.raises(ValueError, match="cache_hits cannot be negative"):
            validate_pipeline_result(result)


class TestValidatePipelineStats:
    """Tests for validate_pipeline_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = PipelineStats(
            total_samples=1000,
            throughput_samples_per_sec=500.0,
            memory_usage_mb=128.0,
            cache_hit_rate=0.5,
        )
        validate_pipeline_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_pipeline_stats(None)  # type: ignore[arg-type]

    def test_negative_total_samples_raises_error(self) -> None:
        """Test that negative total_samples raises ValueError."""
        stats = PipelineStats(
            total_samples=-1,
            throughput_samples_per_sec=500.0,
            memory_usage_mb=128.0,
            cache_hit_rate=0.5,
        )
        with pytest.raises(ValueError, match="total_samples cannot be negative"):
            validate_pipeline_stats(stats)

    def test_negative_throughput_raises_error(self) -> None:
        """Test that negative throughput raises ValueError."""
        stats = PipelineStats(
            total_samples=1000,
            throughput_samples_per_sec=-1.0,
            memory_usage_mb=128.0,
            cache_hit_rate=0.5,
        )
        with pytest.raises(
            ValueError, match="throughput_samples_per_sec cannot be negative"
        ):
            validate_pipeline_stats(stats)

    def test_negative_memory_raises_error(self) -> None:
        """Test that negative memory_usage raises ValueError."""
        stats = PipelineStats(
            total_samples=1000,
            throughput_samples_per_sec=500.0,
            memory_usage_mb=-1.0,
            cache_hit_rate=0.5,
        )
        with pytest.raises(ValueError, match="memory_usage_mb cannot be negative"):
            validate_pipeline_stats(stats)

    def test_invalid_cache_hit_rate_raises_error(self) -> None:
        """Test that invalid cache_hit_rate raises ValueError."""
        stats = PipelineStats(
            total_samples=1000,
            throughput_samples_per_sec=500.0,
            memory_usage_mb=128.0,
            cache_hit_rate=1.5,
        )
        with pytest.raises(ValueError, match="cache_hit_rate must be between 0 and 1"):
            validate_pipeline_stats(stats)

    def test_negative_cache_hit_rate_raises_error(self) -> None:
        """Test that negative cache_hit_rate raises ValueError."""
        stats = PipelineStats(
            total_samples=1000,
            throughput_samples_per_sec=500.0,
            memory_usage_mb=128.0,
            cache_hit_rate=-0.1,
        )
        with pytest.raises(ValueError, match="cache_hit_rate must be between 0 and 1"):
            validate_pipeline_stats(stats)


class TestCreateStageConfig:
    """Tests for create_stage_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_stage_config()
        assert config.stage_type == PipelineStage.MAP
        assert config.function_name == "identity"
        assert config.params == {}
        assert config.cache == CacheStrategy.NONE

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_stage_config(
            stage_type="filter",
            function_name="is_valid",
            params={"threshold": 0.5},
            cache="memory",
        )
        assert config.stage_type == PipelineStage.FILTER
        assert config.function_name == "is_valid"
        assert config.params == {"threshold": 0.5}
        assert config.cache == CacheStrategy.MEMORY

    def test_invalid_stage_type_raises_error(self) -> None:
        """Test that invalid stage_type raises ValueError."""
        with pytest.raises(ValueError, match="stage_type must be one of"):
            create_stage_config(stage_type="invalid")

    def test_invalid_cache_raises_error(self) -> None:
        """Test that invalid cache raises ValueError."""
        with pytest.raises(ValueError, match="cache must be one of"):
            create_stage_config(cache="invalid")

    def test_empty_function_name_raises_error(self) -> None:
        """Test that empty function_name raises ValueError."""
        with pytest.raises(ValueError, match="function_name cannot be empty"):
            create_stage_config(function_name="")


class TestCreatePipelineConfig:
    """Tests for create_pipeline_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_pipeline_config()
        assert len(config.stages) == 1
        assert config.execution_mode == ExecutionMode.EAGER
        assert config.num_workers == 1
        assert config.prefetch_factor == 2

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        stage = create_stage_config()
        config = create_pipeline_config(
            stages=(stage,),
            execution_mode="parallel",
            num_workers=4,
            prefetch_factor=4,
        )
        assert config.execution_mode == ExecutionMode.PARALLEL
        assert config.num_workers == 4
        assert config.prefetch_factor == 4

    def test_invalid_execution_mode_raises_error(self) -> None:
        """Test that invalid execution_mode raises ValueError."""
        with pytest.raises(ValueError, match="execution_mode must be one of"):
            create_pipeline_config(execution_mode="invalid")

    def test_zero_workers_raises_error(self) -> None:
        """Test that zero num_workers raises ValueError."""
        with pytest.raises(ValueError, match="num_workers must be positive"):
            create_pipeline_config(num_workers=0)

    def test_negative_prefetch_raises_error(self) -> None:
        """Test that negative prefetch_factor raises ValueError."""
        with pytest.raises(ValueError, match="prefetch_factor cannot be negative"):
            create_pipeline_config(prefetch_factor=-1)


class TestCreatePipelineResult:
    """Tests for create_pipeline_result function."""

    def test_default_values(self) -> None:
        """Test default result values."""
        result = create_pipeline_result()
        assert result.num_processed == 0
        assert result.num_filtered == 0
        assert result.processing_time_seconds == pytest.approx(0.0)
        assert result.cache_hits == 0

    def test_custom_values(self) -> None:
        """Test custom result values."""
        result = create_pipeline_result(
            num_processed=1000,
            num_filtered=50,
            processing_time_seconds=2.5,
            cache_hits=100,
        )
        assert result.num_processed == 1000
        assert result.num_filtered == 50
        assert result.processing_time_seconds == pytest.approx(2.5)
        assert result.cache_hits == 100

    def test_negative_num_processed_raises_error(self) -> None:
        """Test that negative num_processed raises ValueError."""
        with pytest.raises(ValueError, match="num_processed cannot be negative"):
            create_pipeline_result(num_processed=-1)


class TestCreatePipelineStats:
    """Tests for create_pipeline_stats function."""

    def test_default_values(self) -> None:
        """Test default stats values."""
        stats = create_pipeline_stats()
        assert stats.total_samples == 0
        assert stats.throughput_samples_per_sec == pytest.approx(0.0)
        assert stats.memory_usage_mb == pytest.approx(0.0)
        assert stats.cache_hit_rate == pytest.approx(0.0)

    def test_custom_values(self) -> None:
        """Test custom stats values."""
        stats = create_pipeline_stats(
            total_samples=10000,
            throughput_samples_per_sec=5000.0,
            memory_usage_mb=256.0,
            cache_hit_rate=0.9,
        )
        assert stats.total_samples == 10000
        assert stats.throughput_samples_per_sec == pytest.approx(5000.0)
        assert stats.memory_usage_mb == pytest.approx(256.0)
        assert stats.cache_hit_rate == pytest.approx(0.9)

    def test_invalid_cache_hit_rate_raises_error(self) -> None:
        """Test that invalid cache_hit_rate raises ValueError."""
        with pytest.raises(ValueError, match="cache_hit_rate must be between 0 and 1"):
            create_pipeline_stats(cache_hit_rate=1.5)


class TestListPipelineStages:
    """Tests for list_pipeline_stages function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        stages = list_pipeline_stages()
        assert isinstance(stages, list)

    def test_contains_expected_stages(self) -> None:
        """Test that list contains expected stages."""
        stages = list_pipeline_stages()
        assert "load" in stages
        assert "transform" in stages
        assert "filter" in stages
        assert "map" in stages
        assert "batch" in stages
        assert "cache" in stages

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        stages = list_pipeline_stages()
        assert stages == sorted(stages)


class TestGetPipelineStage:
    """Tests for get_pipeline_stage function."""

    def test_valid_load(self) -> None:
        """Test getting LOAD stage."""
        assert get_pipeline_stage("load") == PipelineStage.LOAD

    def test_valid_map(self) -> None:
        """Test getting MAP stage."""
        assert get_pipeline_stage("map") == PipelineStage.MAP

    def test_invalid_stage_raises_error(self) -> None:
        """Test that invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="invalid pipeline stage"):
            get_pipeline_stage("invalid")


class TestListExecutionModes:
    """Tests for list_execution_modes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        modes = list_execution_modes()
        assert isinstance(modes, list)

    def test_contains_expected_modes(self) -> None:
        """Test that list contains expected modes."""
        modes = list_execution_modes()
        assert "eager" in modes
        assert "lazy" in modes
        assert "parallel" in modes
        assert "distributed" in modes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        modes = list_execution_modes()
        assert modes == sorted(modes)


class TestGetExecutionMode:
    """Tests for get_execution_mode function."""

    def test_valid_eager(self) -> None:
        """Test getting EAGER mode."""
        assert get_execution_mode("eager") == ExecutionMode.EAGER

    def test_valid_parallel(self) -> None:
        """Test getting PARALLEL mode."""
        assert get_execution_mode("parallel") == ExecutionMode.PARALLEL

    def test_invalid_mode_raises_error(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid execution mode"):
            get_execution_mode("invalid")


class TestListCacheStrategies:
    """Tests for list_cache_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_cache_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_cache_strategies()
        assert "none" in strategies
        assert "memory" in strategies
        assert "disk" in strategies
        assert "hybrid" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_cache_strategies()
        assert strategies == sorted(strategies)


class TestGetCacheStrategy:
    """Tests for get_cache_strategy function."""

    def test_valid_memory(self) -> None:
        """Test getting MEMORY strategy."""
        assert get_cache_strategy("memory") == CacheStrategy.MEMORY

    def test_valid_disk(self) -> None:
        """Test getting DISK strategy."""
        assert get_cache_strategy("disk") == CacheStrategy.DISK

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid cache strategy"):
            get_cache_strategy("invalid")


class TestBuildPipeline:
    """Tests for build_pipeline function."""

    def test_default_pipeline(self) -> None:
        """Test building default pipeline."""
        pipeline = build_pipeline()
        assert len(pipeline.stages) == 1
        assert pipeline.execution_mode == ExecutionMode.EAGER

    def test_custom_pipeline(self) -> None:
        """Test building custom pipeline."""
        pipeline = build_pipeline(
            stages=[
                ("load", "load_dataset", {"name": "squad"}),
                ("map", "tokenize", {"max_length": 512}),
                ("filter", "is_valid", None),
            ],
            execution_mode="parallel",
            num_workers=4,
        )
        assert len(pipeline.stages) == 3
        assert pipeline.stages[0].stage_type == PipelineStage.LOAD
        assert pipeline.stages[1].stage_type == PipelineStage.MAP
        assert pipeline.stages[2].stage_type == PipelineStage.FILTER
        assert pipeline.num_workers == 4

    def test_invalid_stage_raises_error(self) -> None:
        """Test that invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="stage_type must be one of"):
            build_pipeline(stages=[("invalid", "fn", None)])


class TestExecutePipeline:
    """Tests for execute_pipeline function."""

    def test_identity_pipeline(self) -> None:
        """Test executing identity pipeline."""
        data = [1, 2, 3, 4, 5]
        processed, result = execute_pipeline(data)
        assert processed == [1, 2, 3, 4, 5]
        assert result.num_processed == 5
        assert result.num_filtered == 0

    def test_filter_pipeline(self) -> None:
        """Test executing filter pipeline."""
        data = [1, 2, 3, 4, 5, 6]
        filter_stage = create_stage_config(
            stage_type="filter",
            function_name="is_even",
        )
        config = create_pipeline_config(stages=(filter_stage,))
        functions = {"is_even": lambda x: x % 2 == 0}

        processed, result = execute_pipeline(data, config, functions)

        assert processed == [2, 4, 6]
        assert result.num_processed == 3
        assert result.num_filtered == 3

    def test_map_pipeline(self) -> None:
        """Test executing map pipeline."""
        data = [1, 2, 3]
        map_stage = create_stage_config(
            stage_type="map",
            function_name="double",
        )
        config = create_pipeline_config(stages=(map_stage,))
        functions = {"double": lambda x: x * 2}

        processed, result = execute_pipeline(data, config, functions)

        assert processed == [2, 4, 6]
        assert result.num_processed == 3

    def test_batch_pipeline(self) -> None:
        """Test executing batch pipeline."""
        data = [1, 2, 3, 4, 5]
        batch_stage = create_stage_config(
            stage_type="batch",
            function_name="batch",
            params={"batch_size": 2},
        )
        config = create_pipeline_config(stages=(batch_stage,))

        processed, _result = execute_pipeline(data, config)

        assert len(processed) == 3
        assert processed[0] == [1, 2]
        assert processed[1] == [3, 4]
        assert processed[2] == [5]

    def test_empty_data(self) -> None:
        """Test executing pipeline with empty data."""
        data: list[int] = []
        processed, result = execute_pipeline(data)
        assert processed == []
        assert result.num_processed == 0

    def test_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be None"):
            execute_pipeline(None)  # type: ignore[arg-type]

    def test_builtin_functions(self) -> None:
        """Test using built-in functions."""
        data = [1, 2, 3]
        map_stage = create_stage_config(
            stage_type="map",
            function_name="to_string",
        )
        config = create_pipeline_config(stages=(map_stage,))

        processed, _ = execute_pipeline(data, config)

        assert processed == ["1", "2", "3"]

    @given(
        items=st.lists(
            st.integers(min_value=0, max_value=100), min_size=1, max_size=50
        ),
    )
    @settings(max_examples=20)
    def test_filter_reduces_or_preserves_count(self, items: list[int]) -> None:
        """Test that filter never increases count."""
        filter_stage = create_stage_config(
            stage_type="filter",
            function_name="is_even",
        )
        config = create_pipeline_config(stages=(filter_stage,))
        functions = {"is_even": lambda x: x % 2 == 0}

        processed, result = execute_pipeline(items, config, functions)

        assert len(processed) <= len(items)
        assert result.num_filtered == len(items) - len(processed)


class TestOptimizePipeline:
    """Tests for optimize_pipeline function."""

    def test_filters_moved_first(self) -> None:
        """Test that filter stages are moved before map stages."""
        map_stage = create_stage_config(stage_type="map", function_name="expensive")
        filter_stage = create_stage_config(
            stage_type="filter", function_name="is_valid"
        )
        config = create_pipeline_config(stages=(map_stage, filter_stage))

        optimized = optimize_pipeline(config)

        assert optimized.stages[0].stage_type == PipelineStage.FILTER
        assert optimized.stages[1].stage_type == PipelineStage.MAP

    def test_transform_stages_get_caching(self) -> None:
        """Test that transform stages get memory caching added."""
        transform_stage = create_stage_config(
            stage_type="transform",
            function_name="tokenize",
            cache="none",
        )
        config = create_pipeline_config(stages=(transform_stage,))

        optimized = optimize_pipeline(config)

        assert optimized.stages[0].cache == CacheStrategy.MEMORY

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            optimize_pipeline(None)  # type: ignore[arg-type]

    def test_zero_sample_size_raises_error(self) -> None:
        """Test that zero sample_size raises ValueError."""
        stage = create_stage_config()
        config = create_pipeline_config(stages=(stage,))
        with pytest.raises(ValueError, match="sample_size must be positive"):
            optimize_pipeline(config, sample_size=0)


class TestEstimatePipelineMemory:
    """Tests for estimate_pipeline_memory function."""

    def test_basic_estimation(self) -> None:
        """Test basic memory estimation."""
        config = create_pipeline_config()
        memory = estimate_pipeline_memory(config, 10000)
        assert memory > 0

    def test_zero_samples_returns_zero(self) -> None:
        """Test that zero samples returns zero memory."""
        config = create_pipeline_config()
        memory = estimate_pipeline_memory(config, 0)
        assert memory == pytest.approx(0.0)

    def test_more_workers_increases_memory(self) -> None:
        """Test that more workers increases memory estimate."""
        stage = create_stage_config()
        config1 = create_pipeline_config(stages=(stage,), num_workers=1)
        config2 = create_pipeline_config(stages=(stage,), num_workers=8)

        memory1 = estimate_pipeline_memory(config1, 10000)
        memory2 = estimate_pipeline_memory(config2, 10000)

        assert memory2 > memory1

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            estimate_pipeline_memory(None, 100)  # type: ignore[arg-type]

    def test_negative_samples_raises_error(self) -> None:
        """Test that negative num_samples raises ValueError."""
        config = create_pipeline_config()
        with pytest.raises(ValueError, match="num_samples cannot be negative"):
            estimate_pipeline_memory(config, -1)

    def test_zero_sample_size_raises_error(self) -> None:
        """Test that zero avg_sample_size_bytes raises ValueError."""
        config = create_pipeline_config()
        with pytest.raises(ValueError, match="avg_sample_size_bytes must be positive"):
            estimate_pipeline_memory(config, 100, avg_sample_size_bytes=0)


class TestValidatePipelineOutput:
    """Tests for validate_pipeline_output function."""

    def test_valid_output(self) -> None:
        """Test validation of valid output."""
        assert validate_pipeline_output([1, 2, 3], expected_type=int) is True

    def test_invalid_type(self) -> None:
        """Test validation with invalid type."""
        assert validate_pipeline_output([1, "2", 3], expected_type=int) is False

    def test_min_length_check(self) -> None:
        """Test min_length validation."""
        assert validate_pipeline_output([1, 2], min_length=3) is False
        assert validate_pipeline_output([1, 2, 3], min_length=3) is True

    def test_max_length_check(self) -> None:
        """Test max_length validation."""
        assert validate_pipeline_output([1, 2, 3, 4], max_length=3) is False
        assert validate_pipeline_output([1, 2, 3], max_length=3) is True

    def test_none_output_raises_error(self) -> None:
        """Test that None output raises ValueError."""
        with pytest.raises(ValueError, match="output cannot be None"):
            validate_pipeline_output(None)  # type: ignore[arg-type]

    def test_negative_min_length_raises_error(self) -> None:
        """Test that negative min_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length cannot be negative"):
            validate_pipeline_output([1, 2, 3], min_length=-1)

    def test_max_less_than_min_raises_error(self) -> None:
        """Test that max_length < min_length raises ValueError."""
        with pytest.raises(
            ValueError, match=r"max_length.*cannot be less than min_length"
        ):
            validate_pipeline_output([1, 2, 3], min_length=5, max_length=3)


class TestFormatPipelineStats:
    """Tests for format_pipeline_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = PipelineStats(
            total_samples=10000,
            throughput_samples_per_sec=5000.0,
            memory_usage_mb=256.0,
            cache_hit_rate=0.85,
        )
        formatted = format_pipeline_stats(stats)

        assert "10,000" in formatted
        assert "5,000.0" in formatted
        assert "256.0" in formatted
        assert "85.0%" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_pipeline_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedPipelineConfig:
    """Tests for get_recommended_pipeline_config function."""

    def test_training_config(self) -> None:
        """Test getting training configuration."""
        config = get_recommended_pipeline_config("training")
        assert config.execution_mode == ExecutionMode.PARALLEL
        assert any(s.cache != CacheStrategy.NONE for s in config.stages)

    def test_inference_config(self) -> None:
        """Test getting inference configuration."""
        config = get_recommended_pipeline_config("inference")
        assert config.execution_mode == ExecutionMode.EAGER
        assert config.num_workers == 1

    def test_preprocessing_config(self) -> None:
        """Test getting preprocessing configuration."""
        config = get_recommended_pipeline_config("preprocessing")
        assert config.execution_mode == ExecutionMode.PARALLEL
        assert config.num_workers >= 2

    def test_streaming_config(self) -> None:
        """Test getting streaming configuration."""
        config = get_recommended_pipeline_config("streaming")
        assert config.execution_mode == ExecutionMode.LAZY

    def test_default_config(self) -> None:
        """Test getting default configuration for unknown use case."""
        config = get_recommended_pipeline_config("unknown")
        assert config.execution_mode == ExecutionMode.EAGER

    def test_empty_use_case_raises_error(self) -> None:
        """Test that empty use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case cannot be empty"):
            get_recommended_pipeline_config("")

    def test_negative_samples_raises_error(self) -> None:
        """Test that negative num_samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples cannot be negative"):
            get_recommended_pipeline_config("training", num_samples=-1)

    def test_worker_scaling(self) -> None:
        """Test that worker count scales with sample count."""
        config1 = get_recommended_pipeline_config("training", num_samples=1000)
        config2 = get_recommended_pipeline_config("training", num_samples=100000)
        assert config2.num_workers >= config1.num_workers


class TestHypothesisPipelineProperties:
    """Property-based tests for pipeline functions."""

    @given(
        num_samples=st.integers(min_value=0, max_value=100000),
        avg_size=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=20)
    def test_memory_estimation_non_negative(
        self, num_samples: int, avg_size: int
    ) -> None:
        """Test that memory estimation is always non-negative."""
        config = create_pipeline_config()
        memory = estimate_pipeline_memory(config, num_samples, avg_size)
        assert memory >= 0.0

    @given(
        total_samples=st.integers(min_value=0, max_value=100000),
        throughput=st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False),
        memory=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        cache_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_stats_creation_roundtrip(
        self,
        total_samples: int,
        throughput: float,
        memory: float,
        cache_rate: float,
    ) -> None:
        """Test that stats can be created and validated."""
        stats = create_pipeline_stats(
            total_samples=total_samples,
            throughput_samples_per_sec=throughput,
            memory_usage_mb=memory,
            cache_hit_rate=cache_rate,
        )
        validate_pipeline_stats(stats)  # Should not raise
