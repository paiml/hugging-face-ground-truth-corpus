"""Data pipeline orchestration for HuggingFace preprocessing workflows.

This module provides utilities for building, executing, and optimizing
data processing pipelines with support for eager, lazy, parallel, and
distributed execution modes.

Examples:
    >>> from hf_gtc.preprocessing.pipeline import PipelineStage, ExecutionMode
    >>> PipelineStage.LOAD.value
    'load'
    >>> ExecutionMode.EAGER.value
    'eager'
    >>> from hf_gtc.preprocessing.pipeline import CacheStrategy
    >>> CacheStrategy.MEMORY.value
    'memory'
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from hf_gtc._validation import validate_not_none


class PipelineStage(Enum):
    """Stages in a data processing pipeline.

    Attributes:
        LOAD: Load data from source.
        TRANSFORM: Apply transformations to data.
        FILTER: Filter data based on criteria.
        MAP: Apply a function to each element.
        BATCH: Batch data into groups.
        CACHE: Cache intermediate results.

    Examples:
        >>> PipelineStage.LOAD.value
        'load'
        >>> PipelineStage.TRANSFORM.value
        'transform'
        >>> PipelineStage.FILTER.value
        'filter'
    """

    LOAD = "load"
    TRANSFORM = "transform"
    FILTER = "filter"
    MAP = "map"
    BATCH = "batch"
    CACHE = "cache"


VALID_PIPELINE_STAGES = frozenset(s.value for s in PipelineStage)


class ExecutionMode(Enum):
    """Execution modes for pipeline processing.

    Attributes:
        EAGER: Execute operations immediately.
        LAZY: Defer execution until results needed.
        PARALLEL: Execute operations in parallel.
        DISTRIBUTED: Distribute execution across nodes.

    Examples:
        >>> ExecutionMode.EAGER.value
        'eager'
        >>> ExecutionMode.LAZY.value
        'lazy'
        >>> ExecutionMode.PARALLEL.value
        'parallel'
    """

    EAGER = "eager"
    LAZY = "lazy"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


VALID_EXECUTION_MODES = frozenset(m.value for m in ExecutionMode)


class CacheStrategy(Enum):
    """Caching strategies for pipeline stages.

    Attributes:
        NONE: No caching.
        MEMORY: Cache in memory.
        DISK: Cache to disk.
        HYBRID: Use memory with disk overflow.

    Examples:
        >>> CacheStrategy.NONE.value
        'none'
        >>> CacheStrategy.MEMORY.value
        'memory'
        >>> CacheStrategy.DISK.value
        'disk'
    """

    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"


VALID_CACHE_STRATEGIES = frozenset(s.value for s in CacheStrategy)


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Configuration for a single pipeline stage.

    Attributes:
        stage_type: Type of pipeline stage.
        function_name: Name of the function to execute.
        params: Parameters for the function.
        cache: Caching strategy for this stage.

    Examples:
        >>> config = StageConfig(
        ...     stage_type=PipelineStage.MAP,
        ...     function_name="tokenize",
        ...     params={"max_length": 512},
        ...     cache=CacheStrategy.MEMORY,
        ... )
        >>> config.stage_type
        <PipelineStage.MAP: 'map'>
        >>> config.function_name
        'tokenize'
    """

    stage_type: PipelineStage
    function_name: str
    params: dict[str, Any]
    cache: CacheStrategy


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for a complete data pipeline.

    Attributes:
        stages: Tuple of stage configurations.
        execution_mode: How to execute the pipeline.
        num_workers: Number of worker processes/threads.
        prefetch_factor: Number of batches to prefetch.

    Examples:
        >>> stage = StageConfig(
        ...     stage_type=PipelineStage.MAP,
        ...     function_name="process",
        ...     params={},
        ...     cache=CacheStrategy.NONE,
        ... )
        >>> config = PipelineConfig(
        ...     stages=(stage,),
        ...     execution_mode=ExecutionMode.EAGER,
        ...     num_workers=4,
        ...     prefetch_factor=2,
        ... )
        >>> config.num_workers
        4
    """

    stages: tuple[StageConfig, ...]
    execution_mode: ExecutionMode
    num_workers: int
    prefetch_factor: int


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Result from pipeline execution.

    Attributes:
        num_processed: Number of samples processed.
        num_filtered: Number of samples filtered out.
        processing_time_seconds: Total processing time.
        cache_hits: Number of cache hits during execution.

    Examples:
        >>> result = PipelineResult(
        ...     num_processed=10000,
        ...     num_filtered=500,
        ...     processing_time_seconds=5.5,
        ...     cache_hits=1000,
        ... )
        >>> result.num_processed
        10000
        >>> result.processing_time_seconds
        5.5
    """

    num_processed: int
    num_filtered: int
    processing_time_seconds: float
    cache_hits: int


@dataclass(frozen=True, slots=True)
class PipelineStats:
    """Statistics from pipeline execution.

    Attributes:
        total_samples: Total number of samples in pipeline.
        throughput_samples_per_sec: Processing throughput.
        memory_usage_mb: Peak memory usage in megabytes.
        cache_hit_rate: Cache hit rate as a fraction (0.0-1.0).

    Examples:
        >>> stats = PipelineStats(
        ...     total_samples=50000,
        ...     throughput_samples_per_sec=10000.0,
        ...     memory_usage_mb=256.0,
        ...     cache_hit_rate=0.85,
        ... )
        >>> stats.throughput_samples_per_sec
        10000.0
        >>> stats.cache_hit_rate
        0.85
    """

    total_samples: int
    throughput_samples_per_sec: float
    memory_usage_mb: float
    cache_hit_rate: float


def validate_stage_config(config: StageConfig) -> None:
    """Validate a stage configuration.

    Args:
        config: StageConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If function_name is empty.

    Examples:
        >>> config = StageConfig(
        ...     stage_type=PipelineStage.MAP,
        ...     function_name="process",
        ...     params={},
        ...     cache=CacheStrategy.NONE,
        ... )
        >>> validate_stage_config(config)  # No error

        >>> validate_stage_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = StageConfig(
        ...     stage_type=PipelineStage.MAP,
        ...     function_name="",
        ...     params={},
        ...     cache=CacheStrategy.NONE,
        ... )
        >>> validate_stage_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: function_name cannot be empty
    """
    validate_not_none(config, "config")

    if not config.function_name:
        msg = "function_name cannot be empty"
        raise ValueError(msg)


def validate_pipeline_config(config: PipelineConfig) -> None:
    """Validate a pipeline configuration.

    Args:
        config: PipelineConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If stages is empty.
        ValueError: If num_workers is not positive.
        ValueError: If prefetch_factor is negative.

    Examples:
        >>> stage = StageConfig(
        ...     stage_type=PipelineStage.MAP,
        ...     function_name="process",
        ...     params={},
        ...     cache=CacheStrategy.NONE,
        ... )
        >>> config = PipelineConfig(
        ...     stages=(stage,),
        ...     execution_mode=ExecutionMode.EAGER,
        ...     num_workers=4,
        ...     prefetch_factor=2,
        ... )
        >>> validate_pipeline_config(config)  # No error

        >>> validate_pipeline_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = PipelineConfig(
        ...     stages=(),
        ...     execution_mode=ExecutionMode.EAGER,
        ...     num_workers=4,
        ...     prefetch_factor=2,
        ... )
        >>> validate_pipeline_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stages cannot be empty
    """
    validate_not_none(config, "config")

    if not config.stages:
        msg = "stages cannot be empty"
        raise ValueError(msg)

    if config.num_workers <= 0:
        msg = f"num_workers must be positive, got {config.num_workers}"
        raise ValueError(msg)

    if config.prefetch_factor < 0:
        msg = f"prefetch_factor cannot be negative, got {config.prefetch_factor}"
        raise ValueError(msg)

    for stage in config.stages:
        validate_stage_config(stage)


def validate_pipeline_result(result: PipelineResult) -> None:
    """Validate a pipeline result.

    Args:
        result: PipelineResult to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If num_processed is negative.
        ValueError: If num_filtered is negative.
        ValueError: If processing_time_seconds is negative.
        ValueError: If cache_hits is negative.

    Examples:
        >>> result = PipelineResult(
        ...     num_processed=100,
        ...     num_filtered=10,
        ...     processing_time_seconds=1.0,
        ...     cache_hits=50,
        ... )
        >>> validate_pipeline_result(result)  # No error

        >>> validate_pipeline_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = PipelineResult(
        ...     num_processed=-1,
        ...     num_filtered=0,
        ...     processing_time_seconds=1.0,
        ...     cache_hits=0,
        ... )
        >>> validate_pipeline_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_processed cannot be negative
    """
    validate_not_none(result, "result")

    if result.num_processed < 0:
        msg = f"num_processed cannot be negative, got {result.num_processed}"
        raise ValueError(msg)

    if result.num_filtered < 0:
        msg = f"num_filtered cannot be negative, got {result.num_filtered}"
        raise ValueError(msg)

    if result.processing_time_seconds < 0:
        msg = (
            f"processing_time_seconds cannot be negative, "
            f"got {result.processing_time_seconds}"
        )
        raise ValueError(msg)

    if result.cache_hits < 0:
        msg = f"cache_hits cannot be negative, got {result.cache_hits}"
        raise ValueError(msg)


def validate_pipeline_stats(stats: PipelineStats) -> None:
    """Validate pipeline statistics.

    Args:
        stats: PipelineStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If total_samples is negative.
        ValueError: If throughput_samples_per_sec is negative.
        ValueError: If memory_usage_mb is negative.
        ValueError: If cache_hit_rate is not in [0, 1].

    Examples:
        >>> stats = PipelineStats(
        ...     total_samples=1000,
        ...     throughput_samples_per_sec=500.0,
        ...     memory_usage_mb=128.0,
        ...     cache_hit_rate=0.5,
        ... )
        >>> validate_pipeline_stats(stats)  # No error

        >>> validate_pipeline_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = PipelineStats(
        ...     total_samples=1000,
        ...     throughput_samples_per_sec=500.0,
        ...     memory_usage_mb=128.0,
        ...     cache_hit_rate=1.5,
        ... )
        >>> validate_pipeline_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: cache_hit_rate must be between 0 and 1
    """
    validate_not_none(stats, "stats")

    if stats.total_samples < 0:
        msg = f"total_samples cannot be negative, got {stats.total_samples}"
        raise ValueError(msg)

    if stats.throughput_samples_per_sec < 0:
        msg = (
            f"throughput_samples_per_sec cannot be negative, "
            f"got {stats.throughput_samples_per_sec}"
        )
        raise ValueError(msg)

    if stats.memory_usage_mb < 0:
        msg = f"memory_usage_mb cannot be negative, got {stats.memory_usage_mb}"
        raise ValueError(msg)

    if not 0.0 <= stats.cache_hit_rate <= 1.0:
        msg = f"cache_hit_rate must be between 0 and 1, got {stats.cache_hit_rate}"
        raise ValueError(msg)


def create_stage_config(
    stage_type: str = "map",
    function_name: str = "identity",
    params: dict[str, Any] | None = None,
    cache: str = "none",
) -> StageConfig:
    """Create a stage configuration.

    Args:
        stage_type: Type of pipeline stage. Defaults to "map".
        function_name: Name of the function. Defaults to "identity".
        params: Function parameters. Defaults to empty dict.
        cache: Caching strategy. Defaults to "none".

    Returns:
        StageConfig with the specified settings.

    Raises:
        ValueError: If stage_type is invalid.
        ValueError: If cache is invalid.
        ValueError: If function_name is empty.

    Examples:
        >>> config = create_stage_config(stage_type="filter", function_name="is_valid")
        >>> config.stage_type
        <PipelineStage.FILTER: 'filter'>

        >>> config = create_stage_config(cache="memory")
        >>> config.cache
        <CacheStrategy.MEMORY: 'memory'>

        >>> create_stage_config(stage_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stage_type must be one of
    """
    if stage_type not in VALID_PIPELINE_STAGES:
        msg = f"stage_type must be one of {VALID_PIPELINE_STAGES}, got '{stage_type}'"
        raise ValueError(msg)

    if cache not in VALID_CACHE_STRATEGIES:
        msg = f"cache must be one of {VALID_CACHE_STRATEGIES}, got '{cache}'"
        raise ValueError(msg)

    config = StageConfig(
        stage_type=PipelineStage(stage_type),
        function_name=function_name,
        params=params if params is not None else {},
        cache=CacheStrategy(cache),
    )
    validate_stage_config(config)
    return config


def create_pipeline_config(
    stages: tuple[StageConfig, ...] | None = None,
    execution_mode: str = "eager",
    num_workers: int = 1,
    prefetch_factor: int = 2,
) -> PipelineConfig:
    """Create a pipeline configuration.

    Args:
        stages: Tuple of stage configurations. Defaults to single identity stage.
        execution_mode: Execution mode. Defaults to "eager".
        num_workers: Number of workers. Defaults to 1.
        prefetch_factor: Prefetch factor. Defaults to 2.

    Returns:
        PipelineConfig with the specified settings.

    Raises:
        ValueError: If execution_mode is invalid.
        ValueError: If num_workers is not positive.
        ValueError: If prefetch_factor is negative.

    Examples:
        >>> config = create_pipeline_config(execution_mode="parallel", num_workers=4)
        >>> config.execution_mode
        <ExecutionMode.PARALLEL: 'parallel'>
        >>> config.num_workers
        4

        >>> create_pipeline_config(execution_mode="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: execution_mode must be one of
    """
    if execution_mode not in VALID_EXECUTION_MODES:
        msg = (
            f"execution_mode must be one of {VALID_EXECUTION_MODES}, "
            f"got '{execution_mode}'"
        )
        raise ValueError(msg)

    effective_stages = stages
    if effective_stages is None:
        effective_stages = (create_stage_config(),)

    config = PipelineConfig(
        stages=effective_stages,
        execution_mode=ExecutionMode(execution_mode),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    validate_pipeline_config(config)
    return config


def create_pipeline_result(
    num_processed: int = 0,
    num_filtered: int = 0,
    processing_time_seconds: float = 0.0,
    cache_hits: int = 0,
) -> PipelineResult:
    """Create a pipeline result.

    Args:
        num_processed: Number of samples processed. Defaults to 0.
        num_filtered: Number of samples filtered. Defaults to 0.
        processing_time_seconds: Processing time. Defaults to 0.0.
        cache_hits: Number of cache hits. Defaults to 0.

    Returns:
        PipelineResult with the specified values.

    Raises:
        ValueError: If any value is negative.

    Examples:
        >>> result = create_pipeline_result(num_processed=1000, num_filtered=50)
        >>> result.num_processed
        1000

        >>> create_pipeline_result(num_processed=-1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_processed cannot be negative
    """
    result = PipelineResult(
        num_processed=num_processed,
        num_filtered=num_filtered,
        processing_time_seconds=processing_time_seconds,
        cache_hits=cache_hits,
    )
    validate_pipeline_result(result)
    return result


def create_pipeline_stats(
    total_samples: int = 0,
    throughput_samples_per_sec: float = 0.0,
    memory_usage_mb: float = 0.0,
    cache_hit_rate: float = 0.0,
) -> PipelineStats:
    """Create pipeline statistics.

    Args:
        total_samples: Total samples in pipeline. Defaults to 0.
        throughput_samples_per_sec: Processing throughput. Defaults to 0.0.
        memory_usage_mb: Memory usage in MB. Defaults to 0.0.
        cache_hit_rate: Cache hit rate (0.0-1.0). Defaults to 0.0.

    Returns:
        PipelineStats with the specified values.

    Raises:
        ValueError: If any value is invalid.

    Examples:
        >>> stats = create_pipeline_stats(
        ...     total_samples=1000,
        ...     throughput_samples_per_sec=500.0,
        ... )
        >>> stats.total_samples
        1000

        >>> create_pipeline_stats(cache_hit_rate=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: cache_hit_rate must be between 0 and 1
    """
    stats = PipelineStats(
        total_samples=total_samples,
        throughput_samples_per_sec=throughput_samples_per_sec,
        memory_usage_mb=memory_usage_mb,
        cache_hit_rate=cache_hit_rate,
    )
    validate_pipeline_stats(stats)
    return stats


def list_pipeline_stages() -> list[str]:
    """List all available pipeline stages.

    Returns:
        Sorted list of pipeline stage names.

    Examples:
        >>> stages = list_pipeline_stages()
        >>> "load" in stages
        True
        >>> "map" in stages
        True
        >>> stages == sorted(stages)
        True
    """
    return sorted(VALID_PIPELINE_STAGES)


def get_pipeline_stage(name: str) -> PipelineStage:
    """Get PipelineStage enum from string name.

    Args:
        name: Name of the pipeline stage.

    Returns:
        Corresponding PipelineStage enum value.

    Raises:
        ValueError: If name is not a valid pipeline stage.

    Examples:
        >>> get_pipeline_stage("load")
        <PipelineStage.LOAD: 'load'>

        >>> get_pipeline_stage("map")
        <PipelineStage.MAP: 'map'>

        >>> get_pipeline_stage("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid pipeline stage: invalid
    """
    if name not in VALID_PIPELINE_STAGES:
        msg = f"invalid pipeline stage: {name}"
        raise ValueError(msg)

    return PipelineStage(name)


def list_execution_modes() -> list[str]:
    """List all available execution modes.

    Returns:
        Sorted list of execution mode names.

    Examples:
        >>> modes = list_execution_modes()
        >>> "eager" in modes
        True
        >>> "parallel" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_EXECUTION_MODES)


def get_execution_mode(name: str) -> ExecutionMode:
    """Get ExecutionMode enum from string name.

    Args:
        name: Name of the execution mode.

    Returns:
        Corresponding ExecutionMode enum value.

    Raises:
        ValueError: If name is not a valid execution mode.

    Examples:
        >>> get_execution_mode("eager")
        <ExecutionMode.EAGER: 'eager'>

        >>> get_execution_mode("parallel")
        <ExecutionMode.PARALLEL: 'parallel'>

        >>> get_execution_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid execution mode: invalid
    """
    if name not in VALID_EXECUTION_MODES:
        msg = f"invalid execution mode: {name}"
        raise ValueError(msg)

    return ExecutionMode(name)


def list_cache_strategies() -> list[str]:
    """List all available cache strategies.

    Returns:
        Sorted list of cache strategy names.

    Examples:
        >>> strategies = list_cache_strategies()
        >>> "memory" in strategies
        True
        >>> "disk" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_CACHE_STRATEGIES)


def get_cache_strategy(name: str) -> CacheStrategy:
    """Get CacheStrategy enum from string name.

    Args:
        name: Name of the cache strategy.

    Returns:
        Corresponding CacheStrategy enum value.

    Raises:
        ValueError: If name is not a valid cache strategy.

    Examples:
        >>> get_cache_strategy("memory")
        <CacheStrategy.MEMORY: 'memory'>

        >>> get_cache_strategy("disk")
        <CacheStrategy.DISK: 'disk'>

        >>> get_cache_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid cache strategy: invalid
    """
    if name not in VALID_CACHE_STRATEGIES:
        msg = f"invalid cache strategy: {name}"
        raise ValueError(msg)

    return CacheStrategy(name)


def build_pipeline(
    stages: Sequence[tuple[str, str, dict[str, Any] | None]] | None = None,
    execution_mode: str = "eager",
    num_workers: int = 1,
) -> PipelineConfig:
    """Build a pipeline from stage specifications.

    Args:
        stages: Sequence of (stage_type, function_name, params) tuples.
            Defaults to a single identity map stage.
        execution_mode: Execution mode. Defaults to "eager".
        num_workers: Number of workers. Defaults to 1.

    Returns:
        PipelineConfig ready for execution.

    Raises:
        ValueError: If stages is None (explicit).
        ValueError: If any stage specification is invalid.

    Examples:
        >>> pipeline = build_pipeline([
        ...     ("load", "load_dataset", {"name": "squad"}),
        ...     ("map", "tokenize", {"max_length": 512}),
        ...     ("filter", "is_valid", None),
        ... ])
        >>> len(pipeline.stages)
        3

        >>> pipeline = build_pipeline()
        >>> len(pipeline.stages)
        1

        >>> build_pipeline([("invalid", "fn", None)])
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stage_type must be one of
    """
    if stages is None:
        stage_configs = (create_stage_config(),)
    else:
        stage_configs = tuple(
            create_stage_config(
                stage_type=stage_type,
                function_name=fn_name,
                params=params,
            )
            for stage_type, fn_name, params in stages
        )

    return create_pipeline_config(
        stages=stage_configs,
        execution_mode=execution_mode,
        num_workers=num_workers,
    )


def execute_pipeline(
    data: Sequence[Any],
    config: PipelineConfig | None = None,
    functions: dict[str, Callable[..., Any]] | None = None,
) -> tuple[list[Any], PipelineResult]:
    """Execute a pipeline on input data.

    Args:
        data: Input data sequence.
        config: Pipeline configuration. Defaults to single identity stage.
        functions: Dictionary mapping function names to callables.
            Defaults to built-in functions.

    Returns:
        Tuple of (processed_data, result).

    Raises:
        ValueError: If data is None.

    Examples:
        >>> data = [1, 2, 3, 4, 5]
        >>> processed, result = execute_pipeline(data)
        >>> len(processed)
        5
        >>> result.num_processed
        5

        >>> data = [1, 2, 3, 4, 5, 6]
        >>> filter_stage = create_stage_config(
        ...     stage_type="filter",
        ...     function_name="is_even",
        ... )
        >>> config = create_pipeline_config(stages=(filter_stage,))
        >>> functions = {"is_even": lambda x: x % 2 == 0}
        >>> processed, result = execute_pipeline(data, config, functions)
        >>> processed
        [2, 4, 6]
        >>> result.num_filtered
        3

        >>> execute_pipeline(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data cannot be None
    """
    if data is None:
        msg = "data cannot be None"
        raise ValueError(msg)

    effective_config = config or create_pipeline_config()
    effective_functions = functions or {}

    # Built-in functions
    builtin_functions: dict[str, Callable[..., Any]] = {
        "identity": lambda x: x,
        "to_string": str,
        "to_int": int,
        "to_float": float,
        "is_truthy": bool,
    }

    all_functions = {**builtin_functions, **effective_functions}

    start_time = time.perf_counter()
    result_data: list[Any] = list(data)
    total_filtered = 0
    cache_hits = 0

    for stage in effective_config.stages:
        fn = all_functions.get(stage.function_name)
        if fn is None:
            # Unknown function, use identity
            fn = lambda x: x  # noqa: E731

        if stage.stage_type == PipelineStage.FILTER:
            before_len = len(result_data)
            result_data = [item for item in result_data if fn(item)]
            total_filtered += before_len - len(result_data)
        elif stage.stage_type == PipelineStage.MAP:
            result_data = [fn(item) for item in result_data]
        elif stage.stage_type == PipelineStage.TRANSFORM:
            result_data = [fn(item, **stage.params) for item in result_data]
        elif stage.stage_type == PipelineStage.BATCH:
            batch_size = stage.params.get("batch_size", 32)
            batched: list[Any] = []
            for i in range(0, len(result_data), batch_size):
                batched.append(result_data[i : i + batch_size])
            result_data = batched
        # LOAD and CACHE stages are no-ops in this simplified implementation

    elapsed = time.perf_counter() - start_time

    pipeline_result = create_pipeline_result(
        num_processed=len(result_data),
        num_filtered=total_filtered,
        processing_time_seconds=elapsed,
        cache_hits=cache_hits,
    )

    return result_data, pipeline_result


def optimize_pipeline(
    config: PipelineConfig,
    sample_size: int = 100,
) -> PipelineConfig:
    """Optimize a pipeline configuration for better performance.

    Analyzes the pipeline stages and applies optimizations such as:
    - Reordering filter stages before map stages
    - Adding caching to expensive stages
    - Adjusting worker count based on stage characteristics

    Args:
        config: Pipeline configuration to optimize.
        sample_size: Sample size for profiling. Defaults to 100.

    Returns:
        Optimized PipelineConfig.

    Raises:
        ValueError: If config is None.
        ValueError: If sample_size is not positive.

    Examples:
        >>> stage1 = create_stage_config(stage_type="map", function_name="expensive")
        >>> stage2 = create_stage_config(stage_type="filter", function_name="is_valid")
        >>> config = create_pipeline_config(stages=(stage1, stage2))
        >>> optimized = optimize_pipeline(config)
        >>> optimized.stages[0].stage_type
        <PipelineStage.FILTER: 'filter'>

        >>> optimize_pipeline(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> stage = create_stage_config()
        >>> config = create_pipeline_config(stages=(stage,))
        >>> optimize_pipeline(config, sample_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sample_size must be positive
    """
    validate_not_none(config, "config")

    if sample_size <= 0:
        msg = f"sample_size must be positive, got {sample_size}"
        raise ValueError(msg)

    # Optimization: move filter stages before map stages where possible
    filters: list[StageConfig] = []
    others: list[StageConfig] = []

    for stage in config.stages:
        if stage.stage_type == PipelineStage.FILTER:
            filters.append(stage)
        else:
            others.append(stage)

    # Reorder: filters first, then others
    optimized_stages = tuple(filters + others)

    # Add caching to transform stages if not already cached
    cached_stages: list[StageConfig] = []
    for stage in optimized_stages:
        if (
            stage.stage_type == PipelineStage.TRANSFORM
            and stage.cache == CacheStrategy.NONE
        ):
            cached_stages.append(
                StageConfig(
                    stage_type=stage.stage_type,
                    function_name=stage.function_name,
                    params=stage.params,
                    cache=CacheStrategy.MEMORY,
                )
            )
        else:
            cached_stages.append(stage)

    return PipelineConfig(
        stages=tuple(cached_stages),
        execution_mode=config.execution_mode,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )


def estimate_pipeline_memory(
    config: PipelineConfig,
    num_samples: int,
    avg_sample_size_bytes: int = 1024,
) -> float:
    """Estimate memory usage for pipeline execution.

    Args:
        config: Pipeline configuration.
        num_samples: Number of samples to process.
        avg_sample_size_bytes: Average size per sample in bytes.

    Returns:
        Estimated memory usage in megabytes.

    Raises:
        ValueError: If config is None.
        ValueError: If num_samples is negative.
        ValueError: If avg_sample_size_bytes is not positive.

    Examples:
        >>> config = create_pipeline_config()
        >>> memory = estimate_pipeline_memory(config, 10000)
        >>> memory > 0
        True

        >>> estimate_pipeline_memory(None, 100)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> config = create_pipeline_config()
        >>> estimate_pipeline_memory(config, -1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_samples cannot be negative
    """
    validate_not_none(config, "config")

    if num_samples < 0:
        msg = f"num_samples cannot be negative, got {num_samples}"
        raise ValueError(msg)

    if avg_sample_size_bytes <= 0:
        msg = f"avg_sample_size_bytes must be positive, got {avg_sample_size_bytes}"
        raise ValueError(msg)

    if num_samples == 0:
        return 0.0

    # Base memory: input data
    base_memory_bytes = num_samples * avg_sample_size_bytes

    # Additional memory per stage
    stage_multiplier = 1.0
    for stage in config.stages:
        if stage.stage_type == PipelineStage.BATCH:
            stage_multiplier += 0.1
        elif stage.stage_type == PipelineStage.TRANSFORM:
            stage_multiplier += 0.5
        elif stage.stage_type == PipelineStage.MAP:
            stage_multiplier += 0.3
        elif stage.cache != CacheStrategy.NONE:
            stage_multiplier += 0.5

    # Worker overhead
    worker_overhead = 1.0 + (config.num_workers - 1) * 0.1

    # Prefetch overhead
    prefetch_overhead = 1.0 + config.prefetch_factor * 0.05

    total_bytes = (
        base_memory_bytes * stage_multiplier * worker_overhead * prefetch_overhead
    )
    return total_bytes / (1024 * 1024)  # Convert to MB


def validate_pipeline_output(
    output: Sequence[Any],
    expected_type: type | None = None,
    min_length: int = 0,
    max_length: int | None = None,
) -> bool:
    """Validate pipeline output meets expectations.

    Args:
        output: Output data to validate.
        expected_type: Expected type for each element. Defaults to None (any).
        min_length: Minimum output length. Defaults to 0.
        max_length: Maximum output length. Defaults to None (unlimited).

    Returns:
        True if output is valid, False otherwise.

    Raises:
        ValueError: If output is None.
        ValueError: If min_length is negative.
        ValueError: If max_length is less than min_length.

    Examples:
        >>> validate_pipeline_output([1, 2, 3], expected_type=int)
        True

        >>> validate_pipeline_output([1, "2", 3], expected_type=int)
        False

        >>> validate_pipeline_output([1, 2], min_length=3)
        False

        >>> validate_pipeline_output(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: output cannot be None
    """
    if output is None:
        msg = "output cannot be None"
        raise ValueError(msg)

    if min_length < 0:
        msg = f"min_length cannot be negative, got {min_length}"
        raise ValueError(msg)

    if max_length is not None and max_length < min_length:
        msg = f"max_length ({max_length}) cannot be less than min_length ({min_length})"
        raise ValueError(msg)

    output_len = len(output)

    if output_len < min_length:
        return False

    if max_length is not None and output_len > max_length:
        return False

    if expected_type is not None:
        for item in output:
            if not isinstance(item, expected_type):
                return False

    return True


def format_pipeline_stats(stats: PipelineStats) -> str:
    """Format pipeline statistics as a human-readable string.

    Args:
        stats: PipelineStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = PipelineStats(
        ...     total_samples=10000,
        ...     throughput_samples_per_sec=5000.0,
        ...     memory_usage_mb=256.0,
        ...     cache_hit_rate=0.85,
        ... )
        >>> formatted = format_pipeline_stats(stats)
        >>> "10,000" in formatted
        True
        >>> "5,000.0" in formatted
        True

        >>> format_pipeline_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        "Pipeline Statistics",
        "=" * 40,
        f"Total samples:         {stats.total_samples:,}",
        f"Throughput:            {stats.throughput_samples_per_sec:,.1f} samples/sec",
        f"Memory usage:          {stats.memory_usage_mb:,.1f} MB",
        f"Cache hit rate:        {stats.cache_hit_rate * 100:.1f}%",
    ]

    return "\n".join(lines)


def get_recommended_pipeline_config(
    use_case: str,
    num_samples: int = 10000,
) -> PipelineConfig:
    """Get recommended pipeline configuration for a use case.

    Args:
        use_case: Type of use case (e.g., "training", "inference", "preprocessing").
        num_samples: Estimated number of samples. Defaults to 10000.

    Returns:
        Recommended PipelineConfig for the use case.

    Raises:
        ValueError: If use_case is empty.
        ValueError: If num_samples is negative.

    Examples:
        >>> pipeline = get_recommended_pipeline_config("training")
        >>> pipeline.execution_mode
        <ExecutionMode.PARALLEL: 'parallel'>

        >>> pipeline = get_recommended_pipeline_config("inference", num_samples=100)
        >>> pipeline.num_workers
        1

        >>> get_recommended_pipeline_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case cannot be empty
    """
    if not use_case:
        msg = "use_case cannot be empty"
        raise ValueError(msg)

    if num_samples < 0:
        msg = f"num_samples cannot be negative, got {num_samples}"
        raise ValueError(msg)

    use_case_lower = use_case.lower()

    if use_case_lower in ("training", "fine-tuning", "pretraining"):
        # Training: parallel execution with caching
        load_stage = create_stage_config(
            stage_type="load",
            function_name="load_dataset",
            cache="disk",
        )
        transform_stage = create_stage_config(
            stage_type="transform",
            function_name="tokenize",
            cache="memory",
        )
        batch_stage = create_stage_config(
            stage_type="batch",
            function_name="batch",
            params={"batch_size": 32},
        )

        # Scale workers based on sample count
        num_workers = min(4, max(1, num_samples // 10000))

        return create_pipeline_config(
            stages=(load_stage, transform_stage, batch_stage),
            execution_mode="parallel",
            num_workers=num_workers,
            prefetch_factor=2,
        )

    elif use_case_lower in ("inference", "prediction", "serving"):
        # Inference: eager execution, minimal overhead
        load_stage = create_stage_config(
            stage_type="load",
            function_name="load_input",
        )
        transform_stage = create_stage_config(
            stage_type="transform",
            function_name="preprocess",
        )

        return create_pipeline_config(
            stages=(load_stage, transform_stage),
            execution_mode="eager",
            num_workers=1,
            prefetch_factor=1,
        )

    elif use_case_lower in ("preprocessing", "etl", "data-preparation"):
        # Preprocessing: parallel with heavy caching
        load_stage = create_stage_config(
            stage_type="load",
            function_name="load_raw",
            cache="disk",
        )
        filter_stage = create_stage_config(
            stage_type="filter",
            function_name="is_valid",
        )
        transform_stage = create_stage_config(
            stage_type="transform",
            function_name="clean",
            cache="memory",
        )
        cache_stage = create_stage_config(
            stage_type="cache",
            function_name="save_cache",
            cache="disk",
        )

        num_workers = min(8, max(2, num_samples // 5000))

        return create_pipeline_config(
            stages=(load_stage, filter_stage, transform_stage, cache_stage),
            execution_mode="parallel",
            num_workers=num_workers,
            prefetch_factor=4,
        )

    elif use_case_lower in ("streaming", "realtime", "online"):
        # Streaming: lazy execution
        load_stage = create_stage_config(
            stage_type="load",
            function_name="stream_input",
        )
        map_stage = create_stage_config(
            stage_type="map",
            function_name="process",
        )

        return create_pipeline_config(
            stages=(load_stage, map_stage),
            execution_mode="lazy",
            num_workers=1,
            prefetch_factor=1,
        )

    else:
        # Default: simple eager execution
        return create_pipeline_config(
            execution_mode="eager",
            num_workers=1,
            prefetch_factor=2,
        )
