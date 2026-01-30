"""Tests for video processing utilities."""

from __future__ import annotations

import pytest

from hf_gtc.multimodal.video import (
    VALID_SAMPLING_STRATEGIES,
    VALID_VIDEO_MODELS,
    VALID_VIDEO_TASKS,
    FrameConfig,
    FrameSamplingStrategy,
    VideoClassificationConfig,
    VideoConfig,
    VideoModelType,
    VideoProcessingConfig,
    VideoStats,
    VideoTask,
    calculate_frame_indices,
    create_frame_config,
    create_video_classification_config,
    create_video_config,
    create_video_processing_config,
    create_video_stats,
    estimate_video_memory,
    format_video_duration,
    get_recommended_video_config,
    get_sampling_strategy,
    get_video_model_type,
    get_video_task,
    list_sampling_strategies,
    list_video_model_types,
    list_video_tasks,
    validate_frame_config,
    validate_video_config,
)


class TestVideoModelType:
    """Tests for VideoModelType enum."""

    def test_videomae_value(self) -> None:
        """Test VIDEOMAE value."""
        assert VideoModelType.VIDEOMAE.value == "videomae"

    def test_timesformer_value(self) -> None:
        """Test TIMESFORMER value."""
        assert VideoModelType.TIMESFORMER.value == "timesformer"

    def test_vivit_value(self) -> None:
        """Test VIVIT value."""
        assert VideoModelType.VIVIT.value == "vivit"

    def test_x3d_value(self) -> None:
        """Test X3D value."""
        assert VideoModelType.X3D.value == "x3d"

    def test_slowfast_value(self) -> None:
        """Test SLOWFAST value."""
        assert VideoModelType.SLOWFAST.value == "slowfast"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_VIDEO_MODELS."""
        for model in VideoModelType:
            assert model.value in VALID_VIDEO_MODELS


class TestVideoTask:
    """Tests for VideoTask enum."""

    def test_classification_value(self) -> None:
        """Test CLASSIFICATION value."""
        assert VideoTask.CLASSIFICATION.value == "classification"

    def test_captioning_value(self) -> None:
        """Test CAPTIONING value."""
        assert VideoTask.CAPTIONING.value == "captioning"

    def test_qa_value(self) -> None:
        """Test QA value."""
        assert VideoTask.QA.value == "qa"

    def test_segmentation_value(self) -> None:
        """Test SEGMENTATION value."""
        assert VideoTask.SEGMENTATION.value == "segmentation"

    def test_tracking_value(self) -> None:
        """Test TRACKING value."""
        assert VideoTask.TRACKING.value == "tracking"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_VIDEO_TASKS."""
        for task in VideoTask:
            assert task.value in VALID_VIDEO_TASKS


class TestFrameSamplingStrategy:
    """Tests for FrameSamplingStrategy enum."""

    def test_uniform_value(self) -> None:
        """Test UNIFORM value."""
        assert FrameSamplingStrategy.UNIFORM.value == "uniform"

    def test_random_value(self) -> None:
        """Test RANDOM value."""
        assert FrameSamplingStrategy.RANDOM.value == "random"

    def test_keyframe_value(self) -> None:
        """Test KEYFRAME value."""
        assert FrameSamplingStrategy.KEYFRAME.value == "keyframe"

    def test_dense_value(self) -> None:
        """Test DENSE value."""
        assert FrameSamplingStrategy.DENSE.value == "dense"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_SAMPLING_STRATEGIES."""
        for strategy in FrameSamplingStrategy:
            assert strategy.value in VALID_SAMPLING_STRATEGIES


class TestVideoConfig:
    """Tests for VideoConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating video config."""
        config = VideoConfig(
            model_type=VideoModelType.VIDEOMAE,
            num_frames=16,
            frame_size=224,
            fps=30.0,
        )
        assert config.model_type == VideoModelType.VIDEOMAE
        assert config.num_frames == 16
        assert config.frame_size == 224
        assert config.fps == 30.0

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = VideoConfig(VideoModelType.VIDEOMAE, 16, 224, 30.0)
        with pytest.raises(AttributeError):
            config.num_frames = 32  # type: ignore[misc]


class TestFrameConfig:
    """Tests for FrameConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating frame config."""
        config = FrameConfig(
            sampling_strategy=FrameSamplingStrategy.UNIFORM,
            start_time=0.0,
            end_time=10.0,
            stride=1,
        )
        assert config.sampling_strategy == FrameSamplingStrategy.UNIFORM
        assert config.start_time == 0.0
        assert config.end_time == 10.0
        assert config.stride == 1

    def test_end_time_none(self) -> None:
        """Test with end_time as None."""
        config = FrameConfig(
            sampling_strategy=FrameSamplingStrategy.DENSE,
            start_time=5.0,
            end_time=None,
            stride=2,
        )
        assert config.end_time is None

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, 0.0, 10.0, 1)
        with pytest.raises(AttributeError):
            config.stride = 2  # type: ignore[misc]


class TestVideoProcessingConfig:
    """Tests for VideoProcessingConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating video processing config."""
        config = VideoProcessingConfig(
            resize_mode="crop",
            normalize=True,
            crop_size=224,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        assert config.resize_mode == "crop"
        assert config.normalize is True
        assert config.crop_size == 224
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = VideoProcessingConfig(
            "crop", True, 224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )
        with pytest.raises(AttributeError):
            config.crop_size = 256  # type: ignore[misc]


class TestVideoClassificationConfig:
    """Tests for VideoClassificationConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating video classification config."""
        config = VideoClassificationConfig(
            num_labels=400,
            label_names=("walking", "running", "jumping"),
            threshold=0.5,
        )
        assert config.num_labels == 400
        assert config.label_names == ("walking", "running", "jumping")
        assert config.threshold == 0.5

    def test_empty_label_names(self) -> None:
        """Test with empty label names."""
        config = VideoClassificationConfig(
            num_labels=10,
            label_names=(),
            threshold=0.7,
        )
        assert config.label_names == ()

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = VideoClassificationConfig(100, ("action",), 0.5)
        with pytest.raises(AttributeError):
            config.threshold = 0.8  # type: ignore[misc]


class TestVideoStats:
    """Tests for VideoStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating video stats."""
        stats = VideoStats(
            duration_seconds=120.0,
            total_frames=3600,
            fps=30.0,
            resolution=(1920, 1080),
        )
        assert stats.duration_seconds == 120.0
        assert stats.total_frames == 3600
        assert stats.fps == 30.0
        assert stats.resolution == (1920, 1080)

    def test_frozen(self) -> None:
        """Test stats is immutable."""
        stats = VideoStats(60.0, 1800, 30.0, (1280, 720))
        with pytest.raises(AttributeError):
            stats.fps = 24.0  # type: ignore[misc]


class TestValidateVideoConfig:
    """Tests for validate_video_config function."""

    def test_valid_config(self) -> None:
        """Test valid config."""
        config = VideoConfig(VideoModelType.VIDEOMAE, 16, 224, 30.0)
        validate_video_config(config)  # Should not raise

    def test_zero_num_frames(self) -> None:
        """Test zero num_frames."""
        config = VideoConfig(VideoModelType.VIDEOMAE, 0, 224, 30.0)
        with pytest.raises(ValueError, match="num_frames must be positive"):
            validate_video_config(config)

    def test_negative_num_frames(self) -> None:
        """Test negative num_frames."""
        config = VideoConfig(VideoModelType.VIDEOMAE, -1, 224, 30.0)
        with pytest.raises(ValueError, match="num_frames must be positive"):
            validate_video_config(config)

    def test_zero_frame_size(self) -> None:
        """Test zero frame_size."""
        config = VideoConfig(VideoModelType.VIDEOMAE, 16, 0, 30.0)
        with pytest.raises(ValueError, match="frame_size must be positive"):
            validate_video_config(config)

    def test_negative_frame_size(self) -> None:
        """Test negative frame_size."""
        config = VideoConfig(VideoModelType.VIDEOMAE, 16, -1, 30.0)
        with pytest.raises(ValueError, match="frame_size must be positive"):
            validate_video_config(config)

    def test_zero_fps(self) -> None:
        """Test zero fps."""
        config = VideoConfig(VideoModelType.VIDEOMAE, 16, 224, 0.0)
        with pytest.raises(ValueError, match="fps must be positive"):
            validate_video_config(config)

    def test_negative_fps(self) -> None:
        """Test negative fps."""
        config = VideoConfig(VideoModelType.VIDEOMAE, 16, 224, -1.0)
        with pytest.raises(ValueError, match="fps must be positive"):
            validate_video_config(config)


class TestValidateFrameConfig:
    """Tests for validate_frame_config function."""

    def test_valid_config(self) -> None:
        """Test valid config."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, 0.0, 10.0, 1)
        validate_frame_config(config)  # Should not raise

    def test_valid_config_no_end_time(self) -> None:
        """Test valid config with no end_time."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, 0.0, None, 1)
        validate_frame_config(config)  # Should not raise

    def test_negative_start_time(self) -> None:
        """Test negative start_time."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, -1.0, 10.0, 1)
        with pytest.raises(ValueError, match="start_time cannot be negative"):
            validate_frame_config(config)

    def test_end_time_less_than_start_time(self) -> None:
        """Test end_time less than start_time."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, 10.0, 5.0, 1)
        with pytest.raises(ValueError, match="end_time must be greater"):
            validate_frame_config(config)

    def test_end_time_equal_start_time(self) -> None:
        """Test end_time equal to start_time."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, 5.0, 5.0, 1)
        with pytest.raises(ValueError, match="end_time must be greater"):
            validate_frame_config(config)

    def test_zero_stride(self) -> None:
        """Test zero stride."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, 0.0, 10.0, 0)
        with pytest.raises(ValueError, match="stride must be positive"):
            validate_frame_config(config)

    def test_negative_stride(self) -> None:
        """Test negative stride."""
        config = FrameConfig(FrameSamplingStrategy.UNIFORM, 0.0, 10.0, -1)
        with pytest.raises(ValueError, match="stride must be positive"):
            validate_frame_config(config)


class TestCreateVideoConfig:
    """Tests for create_video_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_video_config()
        assert config.model_type == VideoModelType.VIDEOMAE
        assert config.num_frames == 16
        assert config.frame_size == 224
        assert config.fps == 30.0

    def test_custom_num_frames(self) -> None:
        """Test custom num_frames."""
        config = create_video_config(num_frames=32)
        assert config.num_frames == 32

    def test_custom_frame_size(self) -> None:
        """Test custom frame_size."""
        config = create_video_config(frame_size=256)
        assert config.frame_size == 256

    def test_custom_fps(self) -> None:
        """Test custom fps."""
        config = create_video_config(fps=24.0)
        assert config.fps == 24.0

    @pytest.mark.parametrize(
        "model_type,expected",
        [
            ("videomae", VideoModelType.VIDEOMAE),
            ("timesformer", VideoModelType.TIMESFORMER),
            ("vivit", VideoModelType.VIVIT),
            ("x3d", VideoModelType.X3D),
            ("slowfast", VideoModelType.SLOWFAST),
        ],
    )
    def test_all_model_types(self, model_type: str, expected: VideoModelType) -> None:
        """Test all valid model types."""
        config = create_video_config(model_type=model_type)
        assert config.model_type == expected

    def test_invalid_model_type(self) -> None:
        """Test invalid model type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_video_config(model_type="invalid")

    def test_invalid_num_frames(self) -> None:
        """Test invalid num_frames."""
        with pytest.raises(ValueError, match="num_frames must be positive"):
            create_video_config(num_frames=0)

    def test_invalid_frame_size(self) -> None:
        """Test invalid frame_size."""
        with pytest.raises(ValueError, match="frame_size must be positive"):
            create_video_config(frame_size=-1)

    def test_invalid_fps(self) -> None:
        """Test invalid fps."""
        with pytest.raises(ValueError, match="fps must be positive"):
            create_video_config(fps=0)


class TestCreateFrameConfig:
    """Tests for create_frame_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_frame_config()
        assert config.sampling_strategy == FrameSamplingStrategy.UNIFORM
        assert config.start_time == 0.0
        assert config.end_time is None
        assert config.stride == 1

    def test_custom_start_time(self) -> None:
        """Test custom start_time."""
        config = create_frame_config(start_time=5.0)
        assert config.start_time == 5.0

    def test_custom_end_time(self) -> None:
        """Test custom end_time."""
        config = create_frame_config(end_time=15.0)
        assert config.end_time == 15.0

    def test_custom_stride(self) -> None:
        """Test custom stride."""
        config = create_frame_config(stride=2)
        assert config.stride == 2

    @pytest.mark.parametrize(
        "strategy,expected",
        [
            ("uniform", FrameSamplingStrategy.UNIFORM),
            ("random", FrameSamplingStrategy.RANDOM),
            ("keyframe", FrameSamplingStrategy.KEYFRAME),
            ("dense", FrameSamplingStrategy.DENSE),
        ],
    )
    def test_all_sampling_strategies(
        self, strategy: str, expected: FrameSamplingStrategy
    ) -> None:
        """Test all valid sampling strategies."""
        config = create_frame_config(sampling_strategy=strategy)
        assert config.sampling_strategy == expected

    def test_invalid_sampling_strategy(self) -> None:
        """Test invalid sampling strategy."""
        with pytest.raises(ValueError, match="sampling_strategy must be one of"):
            create_frame_config(sampling_strategy="invalid")

    def test_invalid_start_time(self) -> None:
        """Test invalid start_time."""
        with pytest.raises(ValueError, match="start_time cannot be negative"):
            create_frame_config(start_time=-1.0)

    def test_invalid_stride(self) -> None:
        """Test invalid stride."""
        with pytest.raises(ValueError, match="stride must be positive"):
            create_frame_config(stride=0)

    def test_end_time_before_start(self) -> None:
        """Test end_time before start_time."""
        with pytest.raises(ValueError, match="end_time must be greater"):
            create_frame_config(start_time=10.0, end_time=5.0)


class TestCreateVideoProcessingConfig:
    """Tests for create_video_processing_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_video_processing_config()
        assert config.resize_mode == "crop"
        assert config.normalize is True
        assert config.crop_size == 224
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)

    def test_custom_crop_size(self) -> None:
        """Test custom crop_size."""
        config = create_video_processing_config(crop_size=256)
        assert config.crop_size == 256

    def test_no_normalize(self) -> None:
        """Test normalize=False."""
        config = create_video_processing_config(normalize=False)
        assert config.normalize is False

    def test_custom_mean(self) -> None:
        """Test custom mean."""
        config = create_video_processing_config(mean=(0.5, 0.5, 0.5))
        assert config.mean == (0.5, 0.5, 0.5)

    def test_custom_std(self) -> None:
        """Test custom std."""
        config = create_video_processing_config(std=(0.5, 0.5, 0.5))
        assert config.std == (0.5, 0.5, 0.5)

    @pytest.mark.parametrize("mode", ["crop", "pad", "stretch"])
    def test_valid_resize_modes(self, mode: str) -> None:
        """Test all valid resize modes."""
        config = create_video_processing_config(resize_mode=mode)
        assert config.resize_mode == mode

    def test_invalid_resize_mode(self) -> None:
        """Test invalid resize mode."""
        with pytest.raises(ValueError, match="resize_mode must be one of"):
            create_video_processing_config(resize_mode="invalid")

    def test_zero_crop_size(self) -> None:
        """Test zero crop_size."""
        with pytest.raises(ValueError, match="crop_size must be positive"):
            create_video_processing_config(crop_size=0)

    def test_negative_crop_size(self) -> None:
        """Test negative crop_size."""
        with pytest.raises(ValueError, match="crop_size must be positive"):
            create_video_processing_config(crop_size=-1)

    def test_invalid_mean_length(self) -> None:
        """Test invalid mean length."""
        with pytest.raises(ValueError, match="mean must have 3 values"):
            create_video_processing_config(mean=(0.5, 0.5))  # type: ignore[arg-type]

    def test_invalid_std_length(self) -> None:
        """Test invalid std length."""
        with pytest.raises(ValueError, match="std must have 3 values"):
            create_video_processing_config(std=(0.5,))  # type: ignore[arg-type]


class TestListVideoModelTypes:
    """Tests for list_video_model_types function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        types = list_video_model_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_video_model_types()
        assert "videomae" in types
        assert "timesformer" in types
        assert "vivit" in types
        assert "x3d" in types
        assert "slowfast" in types

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        types = list_video_model_types()
        assert types == sorted(types)


class TestListVideoTasks:
    """Tests for list_video_tasks function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        tasks = list_video_tasks()
        assert isinstance(tasks, list)

    def test_contains_expected_tasks(self) -> None:
        """Test contains expected tasks."""
        tasks = list_video_tasks()
        assert "classification" in tasks
        assert "captioning" in tasks
        assert "qa" in tasks
        assert "segmentation" in tasks
        assert "tracking" in tasks

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        tasks = list_video_tasks()
        assert tasks == sorted(tasks)


class TestListSamplingStrategies:
    """Tests for list_sampling_strategies function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        strategies = list_sampling_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test contains expected strategies."""
        strategies = list_sampling_strategies()
        assert "uniform" in strategies
        assert "random" in strategies
        assert "keyframe" in strategies
        assert "dense" in strategies

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        strategies = list_sampling_strategies()
        assert strategies == sorted(strategies)


class TestGetVideoModelType:
    """Tests for get_video_model_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("videomae", VideoModelType.VIDEOMAE),
            ("timesformer", VideoModelType.TIMESFORMER),
            ("vivit", VideoModelType.VIVIT),
            ("x3d", VideoModelType.X3D),
            ("slowfast", VideoModelType.SLOWFAST),
        ],
    )
    def test_valid_model_types(self, name: str, expected: VideoModelType) -> None:
        """Test getting valid model types."""
        assert get_video_model_type(name) == expected

    def test_invalid_model_type(self) -> None:
        """Test invalid model type."""
        with pytest.raises(ValueError, match="model type must be one of"):
            get_video_model_type("invalid")


class TestGetVideoTask:
    """Tests for get_video_task function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("classification", VideoTask.CLASSIFICATION),
            ("captioning", VideoTask.CAPTIONING),
            ("qa", VideoTask.QA),
            ("segmentation", VideoTask.SEGMENTATION),
            ("tracking", VideoTask.TRACKING),
        ],
    )
    def test_valid_tasks(self, name: str, expected: VideoTask) -> None:
        """Test getting valid tasks."""
        assert get_video_task(name) == expected

    def test_invalid_task(self) -> None:
        """Test invalid task."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_video_task("invalid")


class TestGetSamplingStrategy:
    """Tests for get_sampling_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("uniform", FrameSamplingStrategy.UNIFORM),
            ("random", FrameSamplingStrategy.RANDOM),
            ("keyframe", FrameSamplingStrategy.KEYFRAME),
            ("dense", FrameSamplingStrategy.DENSE),
        ],
    )
    def test_valid_strategies(self, name: str, expected: FrameSamplingStrategy) -> None:
        """Test getting valid strategies."""
        assert get_sampling_strategy(name) == expected

    def test_invalid_strategy(self) -> None:
        """Test invalid strategy."""
        with pytest.raises(ValueError, match="sampling strategy must be one of"):
            get_sampling_strategy("invalid")


class TestCalculateFrameIndices:
    """Tests for calculate_frame_indices function."""

    def test_uniform_basic(self) -> None:
        """Test uniform sampling basic case."""
        indices = calculate_frame_indices(100, 10, FrameSamplingStrategy.UNIFORM)
        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_uniform_single_frame(self) -> None:
        """Test uniform sampling with single frame."""
        indices = calculate_frame_indices(100, 1, FrameSamplingStrategy.UNIFORM)
        assert len(indices) == 1
        assert indices[0] == 50  # Middle frame

    def test_uniform_all_frames(self) -> None:
        """Test uniform sampling with all frames."""
        indices = calculate_frame_indices(10, 10, FrameSamplingStrategy.UNIFORM)
        assert len(indices) == 10
        assert indices == tuple(range(10))

    def test_dense_basic(self) -> None:
        """Test dense sampling basic case."""
        indices = calculate_frame_indices(100, 5, FrameSamplingStrategy.DENSE, stride=2)
        assert len(indices) == 5
        assert indices == (0, 2, 4, 6, 8)

    def test_dense_stride_one(self) -> None:
        """Test dense sampling with stride 1."""
        indices = calculate_frame_indices(100, 5, FrameSamplingStrategy.DENSE, stride=1)
        assert len(indices) == 5
        assert indices == (0, 1, 2, 3, 4)

    def test_keyframe_basic(self) -> None:
        """Test keyframe sampling basic case."""
        indices = calculate_frame_indices(100, 10, FrameSamplingStrategy.KEYFRAME)
        assert len(indices) == 10
        assert indices[0] == 0

    def test_keyframe_single_frame(self) -> None:
        """Test keyframe sampling with single frame."""
        indices = calculate_frame_indices(100, 1, FrameSamplingStrategy.KEYFRAME)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_random_basic(self) -> None:
        """Test random sampling basic case."""
        indices = calculate_frame_indices(100, 5, FrameSamplingStrategy.RANDOM)
        assert len(indices) <= 5  # May have duplicates
        for idx in indices:
            assert 0 <= idx < 100

    def test_random_deterministic(self) -> None:
        """Test random sampling is deterministic."""
        indices1 = calculate_frame_indices(100, 5, FrameSamplingStrategy.RANDOM)
        indices2 = calculate_frame_indices(100, 5, FrameSamplingStrategy.RANDOM)
        assert indices1 == indices2

    def test_zero_total_frames(self) -> None:
        """Test zero total_frames."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            calculate_frame_indices(0, 10, FrameSamplingStrategy.UNIFORM)

    def test_negative_total_frames(self) -> None:
        """Test negative total_frames."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            calculate_frame_indices(-1, 10, FrameSamplingStrategy.UNIFORM)

    def test_zero_num_frames(self) -> None:
        """Test zero num_frames."""
        with pytest.raises(ValueError, match="num_frames must be positive"):
            calculate_frame_indices(100, 0, FrameSamplingStrategy.UNIFORM)

    def test_negative_num_frames(self) -> None:
        """Test negative num_frames."""
        with pytest.raises(ValueError, match="num_frames must be positive"):
            calculate_frame_indices(100, -1, FrameSamplingStrategy.UNIFORM)

    def test_num_frames_exceeds_total(self) -> None:
        """Test num_frames exceeds total_frames."""
        with pytest.raises(ValueError, match="num_frames cannot exceed total_frames"):
            calculate_frame_indices(5, 10, FrameSamplingStrategy.UNIFORM)

    def test_zero_stride(self) -> None:
        """Test zero stride."""
        with pytest.raises(ValueError, match="stride must be positive"):
            calculate_frame_indices(100, 10, FrameSamplingStrategy.DENSE, stride=0)

    def test_negative_stride(self) -> None:
        """Test negative stride."""
        with pytest.raises(ValueError, match="stride must be positive"):
            calculate_frame_indices(100, 10, FrameSamplingStrategy.DENSE, stride=-1)


class TestEstimateVideoMemory:
    """Tests for estimate_video_memory function."""

    def test_basic_calculation(self) -> None:
        """Test basic memory calculation."""
        mem = estimate_video_memory(16, 224)
        expected = 16 * 224 * 224 * 3 * 4 * 1
        assert mem == expected

    def test_batch_size(self) -> None:
        """Test batch_size affects memory."""
        mem_1 = estimate_video_memory(16, 224, batch_size=1)
        mem_2 = estimate_video_memory(16, 224, batch_size=2)
        assert mem_2 == mem_1 * 2

    def test_custom_channels(self) -> None:
        """Test custom channels."""
        mem_3 = estimate_video_memory(16, 224, channels=3)
        mem_1 = estimate_video_memory(16, 224, channels=1)
        assert mem_3 == mem_1 * 3

    def test_custom_dtype_bytes(self) -> None:
        """Test custom dtype_bytes."""
        mem_fp32 = estimate_video_memory(16, 224, dtype_bytes=4)
        mem_fp16 = estimate_video_memory(16, 224, dtype_bytes=2)
        assert mem_fp32 == mem_fp16 * 2

    def test_zero_num_frames(self) -> None:
        """Test zero num_frames."""
        with pytest.raises(ValueError, match="num_frames must be positive"):
            estimate_video_memory(0, 224)

    def test_negative_num_frames(self) -> None:
        """Test negative num_frames."""
        with pytest.raises(ValueError, match="num_frames must be positive"):
            estimate_video_memory(-1, 224)

    def test_zero_frame_size(self) -> None:
        """Test zero frame_size."""
        with pytest.raises(ValueError, match="frame_size must be positive"):
            estimate_video_memory(16, 0)

    def test_negative_frame_size(self) -> None:
        """Test negative frame_size."""
        with pytest.raises(ValueError, match="frame_size must be positive"):
            estimate_video_memory(16, -1)

    def test_zero_channels(self) -> None:
        """Test zero channels."""
        with pytest.raises(ValueError, match="channels must be positive"):
            estimate_video_memory(16, 224, channels=0)

    def test_zero_dtype_bytes(self) -> None:
        """Test zero dtype_bytes."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            estimate_video_memory(16, 224, dtype_bytes=0)

    def test_zero_batch_size(self) -> None:
        """Test zero batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_video_memory(16, 224, batch_size=0)


class TestCreateVideoClassificationConfig:
    """Tests for create_video_classification_config function."""

    def test_basic_config(self) -> None:
        """Test basic config creation."""
        config = create_video_classification_config(num_labels=400)
        assert config.num_labels == 400
        assert config.label_names == ()
        assert config.threshold == 0.5

    def test_with_label_names(self) -> None:
        """Test with label names."""
        config = create_video_classification_config(
            num_labels=3, label_names=("walking", "running", "jumping")
        )
        assert config.label_names == ("walking", "running", "jumping")

    def test_custom_threshold(self) -> None:
        """Test custom threshold."""
        config = create_video_classification_config(num_labels=10, threshold=0.7)
        assert config.threshold == 0.7

    def test_zero_num_labels(self) -> None:
        """Test zero num_labels."""
        with pytest.raises(ValueError, match="num_labels must be positive"):
            create_video_classification_config(num_labels=0)

    def test_negative_num_labels(self) -> None:
        """Test negative num_labels."""
        with pytest.raises(ValueError, match="num_labels must be positive"):
            create_video_classification_config(num_labels=-1)

    def test_threshold_above_one(self) -> None:
        """Test threshold above 1.0."""
        with pytest.raises(ValueError, match=r"threshold must be between 0\.0"):
            create_video_classification_config(num_labels=10, threshold=1.5)

    def test_threshold_below_zero(self) -> None:
        """Test threshold below 0.0."""
        with pytest.raises(ValueError, match=r"threshold must be between 0\.0"):
            create_video_classification_config(num_labels=10, threshold=-0.1)

    def test_threshold_boundary_zero(self) -> None:
        """Test threshold at 0.0 boundary."""
        config = create_video_classification_config(num_labels=10, threshold=0.0)
        assert config.threshold == 0.0

    def test_threshold_boundary_one(self) -> None:
        """Test threshold at 1.0 boundary."""
        config = create_video_classification_config(num_labels=10, threshold=1.0)
        assert config.threshold == 1.0

    def test_label_names_exceed_num_labels(self) -> None:
        """Test label_names exceeds num_labels."""
        with pytest.raises(ValueError, match="label_names length cannot exceed"):
            create_video_classification_config(
                num_labels=2, label_names=("a", "b", "c")
            )


class TestCreateVideoStats:
    """Tests for create_video_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic stats creation."""
        stats = create_video_stats(60.0, 1800, 30.0, (1920, 1080))
        assert stats.duration_seconds == 60.0
        assert stats.total_frames == 1800
        assert stats.fps == 30.0
        assert stats.resolution == (1920, 1080)

    def test_zero_duration(self) -> None:
        """Test zero duration."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            create_video_stats(0, 1800, 30.0, (1920, 1080))

    def test_negative_duration(self) -> None:
        """Test negative duration."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            create_video_stats(-1.0, 1800, 30.0, (1920, 1080))

    def test_zero_total_frames(self) -> None:
        """Test zero total_frames."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            create_video_stats(60.0, 0, 30.0, (1920, 1080))

    def test_negative_total_frames(self) -> None:
        """Test negative total_frames."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            create_video_stats(60.0, -1, 30.0, (1920, 1080))

    def test_zero_fps(self) -> None:
        """Test zero fps."""
        with pytest.raises(ValueError, match="fps must be positive"):
            create_video_stats(60.0, 1800, 0.0, (1920, 1080))

    def test_negative_fps(self) -> None:
        """Test negative fps."""
        with pytest.raises(ValueError, match="fps must be positive"):
            create_video_stats(60.0, 1800, -1.0, (1920, 1080))

    def test_zero_width(self) -> None:
        """Test zero width in resolution."""
        with pytest.raises(ValueError, match="resolution dimensions must be positive"):
            create_video_stats(60.0, 1800, 30.0, (0, 1080))

    def test_zero_height(self) -> None:
        """Test zero height in resolution."""
        with pytest.raises(ValueError, match="resolution dimensions must be positive"):
            create_video_stats(60.0, 1800, 30.0, (1920, 0))

    def test_negative_width(self) -> None:
        """Test negative width in resolution."""
        with pytest.raises(ValueError, match="resolution dimensions must be positive"):
            create_video_stats(60.0, 1800, 30.0, (-1, 1080))

    def test_negative_height(self) -> None:
        """Test negative height in resolution."""
        with pytest.raises(ValueError, match="resolution dimensions must be positive"):
            create_video_stats(60.0, 1800, 30.0, (1920, -1))


class TestGetRecommendedVideoConfig:
    """Tests for get_recommended_video_config function."""

    def test_videomae_config(self) -> None:
        """Test VideoMAE recommended config."""
        config = get_recommended_video_config(VideoModelType.VIDEOMAE)
        assert config.model_type == VideoModelType.VIDEOMAE
        assert config.num_frames == 16
        assert config.frame_size == 224
        assert config.fps == 30.0

    def test_timesformer_config(self) -> None:
        """Test TimeSformer recommended config."""
        config = get_recommended_video_config(VideoModelType.TIMESFORMER)
        assert config.model_type == VideoModelType.TIMESFORMER
        assert config.num_frames == 8
        assert config.frame_size == 224

    def test_vivit_config(self) -> None:
        """Test ViViT recommended config."""
        config = get_recommended_video_config(VideoModelType.VIVIT)
        assert config.model_type == VideoModelType.VIVIT
        assert config.num_frames == 32
        assert config.frame_size == 224

    def test_x3d_config(self) -> None:
        """Test X3D recommended config."""
        config = get_recommended_video_config(VideoModelType.X3D)
        assert config.model_type == VideoModelType.X3D
        assert config.num_frames == 16
        assert config.frame_size == 182

    def test_slowfast_config(self) -> None:
        """Test SlowFast recommended config."""
        config = get_recommended_video_config(VideoModelType.SLOWFAST)
        assert config.model_type == VideoModelType.SLOWFAST
        assert config.num_frames == 32
        assert config.frame_size == 256

    def test_all_model_types_have_config(self) -> None:
        """Test all model types have recommended config."""
        for model_type in VideoModelType:
            config = get_recommended_video_config(model_type)
            assert config.model_type == model_type


class TestFormatVideoDuration:
    """Tests for format_video_duration function."""

    def test_seconds_only(self) -> None:
        """Test formatting seconds only."""
        assert format_video_duration(45.0) == "00:00:45"

    def test_minutes_and_seconds(self) -> None:
        """Test formatting minutes and seconds."""
        assert format_video_duration(90.0) == "00:01:30"

    def test_hours_minutes_seconds(self) -> None:
        """Test formatting hours, minutes, and seconds."""
        assert format_video_duration(3661.5) == "01:01:01"

    def test_zero_seconds(self) -> None:
        """Test zero seconds."""
        assert format_video_duration(0.0) == "00:00:00"

    def test_exactly_one_hour(self) -> None:
        """Test exactly one hour."""
        assert format_video_duration(3600.0) == "01:00:00"

    def test_large_duration(self) -> None:
        """Test large duration (over 99 hours)."""
        # 100 hours = 360000 seconds
        assert format_video_duration(360000.0) == "100:00:00"

    def test_fractional_seconds_truncated(self) -> None:
        """Test fractional seconds are truncated."""
        assert format_video_duration(59.9) == "00:00:59"

    def test_negative_seconds(self) -> None:
        """Test negative seconds."""
        with pytest.raises(ValueError, match="seconds cannot be negative"):
            format_video_duration(-1)


class TestConstantsValid:
    """Tests for module constants."""

    def test_valid_video_models_frozenset(self) -> None:
        """Test VALID_VIDEO_MODELS is a frozenset."""
        assert isinstance(VALID_VIDEO_MODELS, frozenset)

    def test_valid_video_tasks_frozenset(self) -> None:
        """Test VALID_VIDEO_TASKS is a frozenset."""
        assert isinstance(VALID_VIDEO_TASKS, frozenset)

    def test_valid_sampling_strategies_frozenset(self) -> None:
        """Test VALID_SAMPLING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_SAMPLING_STRATEGIES, frozenset)

    def test_valid_video_models_count(self) -> None:
        """Test VALID_VIDEO_MODELS has expected count."""
        assert len(VALID_VIDEO_MODELS) == len(VideoModelType)

    def test_valid_video_tasks_count(self) -> None:
        """Test VALID_VIDEO_TASKS has expected count."""
        assert len(VALID_VIDEO_TASKS) == len(VideoTask)

    def test_valid_sampling_strategies_count(self) -> None:
        """Test VALID_SAMPLING_STRATEGIES has expected count."""
        assert len(VALID_SAMPLING_STRATEGIES) == len(FrameSamplingStrategy)
