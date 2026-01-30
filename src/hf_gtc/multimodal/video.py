"""Video processing utilities for HuggingFace models.

This module provides functions for video understanding models,
frame sampling, and video inference configurations.

Examples:
    >>> from hf_gtc.multimodal.video import create_video_config
    >>> config = create_video_config(num_frames=16, frame_size=224)
    >>> config.num_frames
    16
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class VideoModelType(Enum):
    """Supported video model types.

    Attributes:
        VIDEOMAE: VideoMAE self-supervised video model.
        TIMESFORMER: TimeSformer divided space-time attention.
        VIVIT: ViViT video vision transformer.
        X3D: X3D efficient video model.
        SLOWFAST: SlowFast two-pathway network.

    Examples:
        >>> VideoModelType.VIDEOMAE.value
        'videomae'
        >>> VideoModelType.TIMESFORMER.value
        'timesformer'
    """

    VIDEOMAE = "videomae"
    TIMESFORMER = "timesformer"
    VIVIT = "vivit"
    X3D = "x3d"
    SLOWFAST = "slowfast"


VALID_VIDEO_MODELS = frozenset(m.value for m in VideoModelType)


class VideoTask(Enum):
    """Supported video tasks.

    Attributes:
        CLASSIFICATION: Video classification.
        CAPTIONING: Video captioning.
        QA: Video question answering.
        SEGMENTATION: Video segmentation.
        TRACKING: Object tracking.

    Examples:
        >>> VideoTask.CLASSIFICATION.value
        'classification'
        >>> VideoTask.QA.value
        'qa'
    """

    CLASSIFICATION = "classification"
    CAPTIONING = "captioning"
    QA = "qa"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"


VALID_VIDEO_TASKS = frozenset(t.value for t in VideoTask)


class FrameSamplingStrategy(Enum):
    """Frame sampling strategies.

    Attributes:
        UNIFORM: Uniformly spaced frames.
        RANDOM: Randomly sampled frames.
        KEYFRAME: Sample only keyframes.
        DENSE: Dense frame sampling.

    Examples:
        >>> FrameSamplingStrategy.UNIFORM.value
        'uniform'
        >>> FrameSamplingStrategy.DENSE.value
        'dense'
    """

    UNIFORM = "uniform"
    RANDOM = "random"
    KEYFRAME = "keyframe"
    DENSE = "dense"


VALID_SAMPLING_STRATEGIES = frozenset(s.value for s in FrameSamplingStrategy)


@dataclass(frozen=True, slots=True)
class VideoConfig:
    """Configuration for video model.

    Attributes:
        model_type: Type of video model.
        num_frames: Number of frames to sample.
        frame_size: Frame size (width and height).
        fps: Target frames per second.

    Examples:
        >>> config = VideoConfig(
        ...     model_type=VideoModelType.VIDEOMAE,
        ...     num_frames=16,
        ...     frame_size=224,
        ...     fps=30.0,
        ... )
        >>> config.num_frames
        16
    """

    model_type: VideoModelType
    num_frames: int
    frame_size: int
    fps: float


@dataclass(frozen=True, slots=True)
class FrameConfig:
    """Configuration for frame sampling.

    Attributes:
        sampling_strategy: Frame sampling strategy.
        start_time: Start time in seconds.
        end_time: End time in seconds (None for entire video).
        stride: Frame stride for dense sampling.

    Examples:
        >>> config = FrameConfig(
        ...     sampling_strategy=FrameSamplingStrategy.UNIFORM,
        ...     start_time=0.0,
        ...     end_time=10.0,
        ...     stride=1,
        ... )
        >>> config.sampling_strategy
        <FrameSamplingStrategy.UNIFORM: 'uniform'>
    """

    sampling_strategy: FrameSamplingStrategy
    start_time: float
    end_time: float | None
    stride: int


@dataclass(frozen=True, slots=True)
class VideoProcessingConfig:
    """Configuration for video preprocessing.

    Attributes:
        resize_mode: Resize mode (crop, pad, stretch).
        normalize: Whether to normalize pixel values.
        crop_size: Crop size after resizing.
        mean: Normalization mean values (R, G, B).
        std: Normalization std values (R, G, B).

    Examples:
        >>> config = VideoProcessingConfig(
        ...     resize_mode="crop",
        ...     normalize=True,
        ...     crop_size=224,
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225),
        ... )
        >>> config.crop_size
        224
    """

    resize_mode: str
    normalize: bool
    crop_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class VideoClassificationConfig:
    """Configuration for video classification.

    Attributes:
        num_labels: Number of classification labels.
        label_names: Tuple of label names.
        threshold: Classification threshold.

    Examples:
        >>> config = VideoClassificationConfig(
        ...     num_labels=400,
        ...     label_names=("walking", "running", "jumping"),
        ...     threshold=0.5,
        ... )
        >>> config.num_labels
        400
    """

    num_labels: int
    label_names: tuple[str, ...]
    threshold: float


@dataclass(frozen=True, slots=True)
class VideoStats:
    """Statistics for a video file.

    Attributes:
        duration_seconds: Video duration in seconds.
        total_frames: Total number of frames.
        fps: Frames per second.
        resolution: Resolution as (width, height).

    Examples:
        >>> stats = VideoStats(
        ...     duration_seconds=120.0,
        ...     total_frames=3600,
        ...     fps=30.0,
        ...     resolution=(1920, 1080),
        ... )
        >>> stats.duration_seconds
        120.0
    """

    duration_seconds: float
    total_frames: int
    fps: float
    resolution: tuple[int, int]


def validate_video_config(config: VideoConfig) -> None:
    """Validate video configuration.

    Args:
        config: Video configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = VideoConfig(
        ...     VideoModelType.VIDEOMAE, 16, 224, 30.0
        ... )
        >>> validate_video_config(config)  # No error

        >>> bad = VideoConfig(VideoModelType.VIDEOMAE, 0, 224, 30.0)
        >>> validate_video_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_frames must be positive
    """
    if config.num_frames <= 0:
        msg = f"num_frames must be positive, got {config.num_frames}"
        raise ValueError(msg)

    if config.frame_size <= 0:
        msg = f"frame_size must be positive, got {config.frame_size}"
        raise ValueError(msg)

    if config.fps <= 0:
        msg = f"fps must be positive, got {config.fps}"
        raise ValueError(msg)


def validate_frame_config(config: FrameConfig) -> None:
    """Validate frame configuration.

    Args:
        config: Frame configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = FrameConfig(
        ...     FrameSamplingStrategy.UNIFORM, 0.0, 10.0, 1
        ... )
        >>> validate_frame_config(config)  # No error

        >>> bad = FrameConfig(FrameSamplingStrategy.UNIFORM, -1.0, 10.0, 1)
        >>> validate_frame_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: start_time cannot be negative
    """
    if config.start_time < 0:
        msg = f"start_time cannot be negative, got {config.start_time}"
        raise ValueError(msg)

    if config.end_time is not None and config.end_time <= config.start_time:
        msg = (
            f"end_time must be greater than start_time, "
            f"got end_time={config.end_time}, start_time={config.start_time}"
        )
        raise ValueError(msg)

    if config.stride <= 0:
        msg = f"stride must be positive, got {config.stride}"
        raise ValueError(msg)


def create_video_config(
    model_type: str = "videomae",
    num_frames: int = 16,
    frame_size: int = 224,
    fps: float = 30.0,
) -> VideoConfig:
    """Create a video configuration.

    Args:
        model_type: Video model type. Defaults to "videomae".
        num_frames: Number of frames. Defaults to 16.
        frame_size: Frame size. Defaults to 224.
        fps: Frames per second. Defaults to 30.0.

    Returns:
        VideoConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_video_config(num_frames=32)
        >>> config.num_frames
        32

        >>> config = create_video_config(model_type="timesformer")
        >>> config.model_type
        <VideoModelType.TIMESFORMER: 'timesformer'>

        >>> create_video_config(num_frames=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_frames must be positive
    """
    if model_type not in VALID_VIDEO_MODELS:
        msg = f"model_type must be one of {VALID_VIDEO_MODELS}, got '{model_type}'"
        raise ValueError(msg)

    config = VideoConfig(
        model_type=VideoModelType(model_type),
        num_frames=num_frames,
        frame_size=frame_size,
        fps=fps,
    )
    validate_video_config(config)
    return config


def create_frame_config(
    sampling_strategy: str = "uniform",
    start_time: float = 0.0,
    end_time: float | None = None,
    stride: int = 1,
) -> FrameConfig:
    """Create a frame sampling configuration.

    Args:
        sampling_strategy: Sampling strategy. Defaults to "uniform".
        start_time: Start time in seconds. Defaults to 0.0.
        end_time: End time in seconds. Defaults to None (entire video).
        stride: Frame stride. Defaults to 1.

    Returns:
        FrameConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_frame_config(start_time=5.0, end_time=15.0)
        >>> config.start_time
        5.0

        >>> config = create_frame_config(sampling_strategy="dense", stride=2)
        >>> config.sampling_strategy
        <FrameSamplingStrategy.DENSE: 'dense'>

        >>> create_frame_config(start_time=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: start_time cannot be negative
    """
    if sampling_strategy not in VALID_SAMPLING_STRATEGIES:
        msg = (
            f"sampling_strategy must be one of {VALID_SAMPLING_STRATEGIES}, "
            f"got '{sampling_strategy}'"
        )
        raise ValueError(msg)

    config = FrameConfig(
        sampling_strategy=FrameSamplingStrategy(sampling_strategy),
        start_time=start_time,
        end_time=end_time,
        stride=stride,
    )
    validate_frame_config(config)
    return config


def create_video_processing_config(
    resize_mode: str = "crop",
    normalize: bool = True,
    crop_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> VideoProcessingConfig:
    """Create a video processing configuration.

    Args:
        resize_mode: Resize mode. Defaults to "crop".
        normalize: Whether to normalize. Defaults to True.
        crop_size: Crop size. Defaults to 224.
        mean: Normalization mean. Defaults to ImageNet mean.
        std: Normalization std. Defaults to ImageNet std.

    Returns:
        VideoProcessingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_video_processing_config(crop_size=256)
        >>> config.crop_size
        256

        >>> config = create_video_processing_config(resize_mode="pad")
        >>> config.resize_mode
        'pad'

        >>> create_video_processing_config(crop_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: crop_size must be positive
    """
    valid_resize_modes = {"crop", "pad", "stretch"}
    if resize_mode not in valid_resize_modes:
        msg = f"resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        raise ValueError(msg)

    if crop_size <= 0:
        msg = f"crop_size must be positive, got {crop_size}"
        raise ValueError(msg)

    if len(mean) != 3:
        msg = f"mean must have 3 values, got {len(mean)}"
        raise ValueError(msg)

    if len(std) != 3:
        msg = f"std must have 3 values, got {len(std)}"
        raise ValueError(msg)

    return VideoProcessingConfig(
        resize_mode=resize_mode,
        normalize=normalize,
        crop_size=crop_size,
        mean=mean,
        std=std,
    )


def list_video_model_types() -> list[str]:
    """List supported video model types.

    Returns:
        Sorted list of video model type names.

    Examples:
        >>> types = list_video_model_types()
        >>> "videomae" in types
        True
        >>> "timesformer" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_VIDEO_MODELS)


def list_video_tasks() -> list[str]:
    """List supported video tasks.

    Returns:
        Sorted list of video task names.

    Examples:
        >>> tasks = list_video_tasks()
        >>> "classification" in tasks
        True
        >>> "qa" in tasks
        True
        >>> tasks == sorted(tasks)
        True
    """
    return sorted(VALID_VIDEO_TASKS)


def list_sampling_strategies() -> list[str]:
    """List supported frame sampling strategies.

    Returns:
        Sorted list of sampling strategy names.

    Examples:
        >>> strategies = list_sampling_strategies()
        >>> "uniform" in strategies
        True
        >>> "dense" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_SAMPLING_STRATEGIES)


def get_video_model_type(name: str) -> VideoModelType:
    """Get video model type from name.

    Args:
        name: Model type name.

    Returns:
        VideoModelType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_video_model_type("videomae")
        <VideoModelType.VIDEOMAE: 'videomae'>

        >>> get_video_model_type("timesformer")
        <VideoModelType.TIMESFORMER: 'timesformer'>

        >>> get_video_model_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model type must be one of ...
    """
    if name not in VALID_VIDEO_MODELS:
        msg = f"model type must be one of {VALID_VIDEO_MODELS}, got '{name}'"
        raise ValueError(msg)
    return VideoModelType(name)


def get_video_task(name: str) -> VideoTask:
    """Get video task from name.

    Args:
        name: Task name.

    Returns:
        VideoTask enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_video_task("classification")
        <VideoTask.CLASSIFICATION: 'classification'>

        >>> get_video_task("qa")
        <VideoTask.QA: 'qa'>

        >>> get_video_task("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of ...
    """
    if name not in VALID_VIDEO_TASKS:
        msg = f"task must be one of {VALID_VIDEO_TASKS}, got '{name}'"
        raise ValueError(msg)
    return VideoTask(name)


def get_sampling_strategy(name: str) -> FrameSamplingStrategy:
    """Get sampling strategy from name.

    Args:
        name: Strategy name.

    Returns:
        FrameSamplingStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_sampling_strategy("uniform")
        <FrameSamplingStrategy.UNIFORM: 'uniform'>

        >>> get_sampling_strategy("dense")
        <FrameSamplingStrategy.DENSE: 'dense'>

        >>> get_sampling_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sampling strategy must be one of ...
    """
    if name not in VALID_SAMPLING_STRATEGIES:
        msg = (
            f"sampling strategy must be one of {VALID_SAMPLING_STRATEGIES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return FrameSamplingStrategy(name)


def calculate_frame_indices(
    total_frames: int,
    num_frames: int,
    strategy: FrameSamplingStrategy,
    stride: int = 1,
) -> tuple[int, ...]:
    """Calculate frame indices based on sampling strategy.

    Args:
        total_frames: Total number of frames in video.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy to use.
        stride: Frame stride for dense sampling. Defaults to 1.

    Returns:
        Tuple of frame indices.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> indices = calculate_frame_indices(100, 10, FrameSamplingStrategy.UNIFORM)
        >>> len(indices)
        10
        >>> indices[0]
        0
        >>> indices[-1]
        99

        >>> dense_strat = FrameSamplingStrategy.DENSE
        >>> indices = calculate_frame_indices(100, 5, dense_strat, stride=2)
        >>> len(indices)
        5
        >>> indices
        (0, 2, 4, 6, 8)

        >>> calculate_frame_indices(0, 10, FrameSamplingStrategy.UNIFORM)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_frames must be positive

        >>> calculate_frame_indices(5, 10, FrameSamplingStrategy.UNIFORM)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_frames cannot exceed total_frames
    """
    if total_frames <= 0:
        msg = f"total_frames must be positive, got {total_frames}"
        raise ValueError(msg)

    if num_frames <= 0:
        msg = f"num_frames must be positive, got {num_frames}"
        raise ValueError(msg)

    if num_frames > total_frames:
        msg = (
            f"num_frames cannot exceed total_frames, "
            f"got num_frames={num_frames}, total_frames={total_frames}"
        )
        raise ValueError(msg)

    if stride <= 0:
        msg = f"stride must be positive, got {stride}"
        raise ValueError(msg)

    if strategy == FrameSamplingStrategy.UNIFORM:
        # Uniformly spaced frames
        if num_frames == 1:
            return (total_frames // 2,)
        step = (total_frames - 1) / (num_frames - 1)
        return tuple(int(i * step) for i in range(num_frames))

    elif strategy == FrameSamplingStrategy.DENSE:
        # Dense sampling with stride
        return tuple(i * stride for i in range(num_frames) if i * stride < total_frames)

    elif strategy == FrameSamplingStrategy.KEYFRAME:
        # Sample at regular intervals, simulating keyframes
        # Typically keyframes are at scene boundaries; we approximate with wider spacing
        if num_frames == 1:
            return (0,)
        step = total_frames // num_frames
        return tuple(i * step for i in range(num_frames))

    elif strategy == FrameSamplingStrategy.RANDOM:
        # For deterministic testing, use evenly distributed "random" indices
        # In real usage, this would use actual random sampling
        import hashlib

        # Use a deterministic hash-based approach for reproducibility
        indices = []
        for i in range(num_frames):
            hash_input = f"{total_frames}_{num_frames}_{i}".encode()
            hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
            idx = hash_val % total_frames
            indices.append(idx)
        return tuple(sorted(set(indices))[:num_frames])

    # Should never reach here due to enum validation
    msg = f"Unknown sampling strategy: {strategy}"
    raise ValueError(msg)


def estimate_video_memory(
    num_frames: int,
    frame_size: int,
    channels: int = 3,
    dtype_bytes: int = 4,
    batch_size: int = 1,
) -> int:
    """Estimate memory usage for video processing.

    Args:
        num_frames: Number of frames.
        frame_size: Frame width and height (square).
        channels: Number of color channels. Defaults to 3.
        dtype_bytes: Bytes per element. Defaults to 4 (FP32).
        batch_size: Batch size. Defaults to 1.

    Returns:
        Estimated memory in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> mem = estimate_video_memory(16, 224)
        >>> mem > 0
        True
        >>> mem == 16 * 224 * 224 * 3 * 4 * 1
        True

        >>> mem = estimate_video_memory(32, 224, batch_size=2)
        >>> mem == 32 * 224 * 224 * 3 * 4 * 2
        True

        >>> estimate_video_memory(0, 224)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_frames must be positive

        >>> estimate_video_memory(16, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: frame_size must be positive
    """
    if num_frames <= 0:
        msg = f"num_frames must be positive, got {num_frames}"
        raise ValueError(msg)

    if frame_size <= 0:
        msg = f"frame_size must be positive, got {frame_size}"
        raise ValueError(msg)

    if channels <= 0:
        msg = f"channels must be positive, got {channels}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    # Memory = batch * frames * height * width * channels * dtype_size
    return batch_size * num_frames * frame_size * frame_size * channels * dtype_bytes


def create_video_classification_config(
    num_labels: int,
    label_names: tuple[str, ...] = (),
    threshold: float = 0.5,
) -> VideoClassificationConfig:
    """Create a video classification configuration.

    Args:
        num_labels: Number of classification labels.
        label_names: Tuple of label names. Defaults to ().
        threshold: Classification threshold. Defaults to 0.5.

    Returns:
        VideoClassificationConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_video_classification_config(
        ...     num_labels=400,
        ...     label_names=("walking", "running"),
        ...     threshold=0.7,
        ... )
        >>> config.num_labels
        400

        >>> create_video_classification_config(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_labels must be positive

        >>> create_video_classification_config(10, threshold=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0.0 and 1.0
    """
    if num_labels <= 0:
        msg = f"num_labels must be positive, got {num_labels}"
        raise ValueError(msg)

    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be between 0.0 and 1.0, got {threshold}"
        raise ValueError(msg)

    if label_names and len(label_names) > num_labels:
        msg = (
            f"label_names length cannot exceed num_labels, "
            f"got {len(label_names)} names for {num_labels} labels"
        )
        raise ValueError(msg)

    return VideoClassificationConfig(
        num_labels=num_labels,
        label_names=label_names,
        threshold=threshold,
    )


def create_video_stats(
    duration_seconds: float,
    total_frames: int,
    fps: float,
    resolution: tuple[int, int],
) -> VideoStats:
    """Create video statistics.

    Args:
        duration_seconds: Video duration in seconds.
        total_frames: Total number of frames.
        fps: Frames per second.
        resolution: Resolution as (width, height).

    Returns:
        VideoStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_video_stats(60.0, 1800, 30.0, (1920, 1080))
        >>> stats.duration_seconds
        60.0

        >>> create_video_stats(0, 1800, 30.0, (1920, 1080))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: duration_seconds must be positive

        >>> create_video_stats(60.0, 1800, 30.0, (0, 1080))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: resolution dimensions must be positive
    """
    if duration_seconds <= 0:
        msg = f"duration_seconds must be positive, got {duration_seconds}"
        raise ValueError(msg)

    if total_frames <= 0:
        msg = f"total_frames must be positive, got {total_frames}"
        raise ValueError(msg)

    if fps <= 0:
        msg = f"fps must be positive, got {fps}"
        raise ValueError(msg)

    if resolution[0] <= 0 or resolution[1] <= 0:
        msg = f"resolution dimensions must be positive, got {resolution}"
        raise ValueError(msg)

    return VideoStats(
        duration_seconds=duration_seconds,
        total_frames=total_frames,
        fps=fps,
        resolution=resolution,
    )


def get_recommended_video_config(model_type: VideoModelType) -> VideoConfig:
    """Get recommended configuration for a video model type.

    Args:
        model_type: Video model type.

    Returns:
        Recommended VideoConfig for the model.

    Examples:
        >>> config = get_recommended_video_config(VideoModelType.VIDEOMAE)
        >>> config.num_frames
        16
        >>> config.frame_size
        224

        >>> config = get_recommended_video_config(VideoModelType.TIMESFORMER)
        >>> config.num_frames
        8
    """
    configs: dict[VideoModelType, VideoConfig] = {
        VideoModelType.VIDEOMAE: VideoConfig(
            model_type=VideoModelType.VIDEOMAE,
            num_frames=16,
            frame_size=224,
            fps=30.0,
        ),
        VideoModelType.TIMESFORMER: VideoConfig(
            model_type=VideoModelType.TIMESFORMER,
            num_frames=8,
            frame_size=224,
            fps=30.0,
        ),
        VideoModelType.VIVIT: VideoConfig(
            model_type=VideoModelType.VIVIT,
            num_frames=32,
            frame_size=224,
            fps=30.0,
        ),
        VideoModelType.X3D: VideoConfig(
            model_type=VideoModelType.X3D,
            num_frames=16,
            frame_size=182,
            fps=30.0,
        ),
        VideoModelType.SLOWFAST: VideoConfig(
            model_type=VideoModelType.SLOWFAST,
            num_frames=32,
            frame_size=256,
            fps=30.0,
        ),
    }
    return configs[model_type]


def format_video_duration(seconds: float) -> str:
    """Format video duration as human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string (HH:MM:SS).

    Raises:
        ValueError: If seconds is negative.

    Examples:
        >>> format_video_duration(3661.5)
        '01:01:01'

        >>> format_video_duration(90.0)
        '00:01:30'

        >>> format_video_duration(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: seconds cannot be negative
    """
    if seconds < 0:
        msg = f"seconds cannot be negative, got {seconds}"
        raise ValueError(msg)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
