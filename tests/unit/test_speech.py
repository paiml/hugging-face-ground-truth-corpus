"""Tests for audio.speech module."""

from __future__ import annotations

import pytest

from hf_gtc.audio.speech import (
    VALID_AUDIO_FORMATS,
    VALID_LANGUAGES,
    VALID_TASKS,
    VALID_WHISPER_SIZES,
    AudioConfig,
    AudioFormat,
    SpeechTask,
    TranscriptionResult,
    TTSConfig,
    WhisperConfig,
    WhisperSize,
    calculate_audio_duration,
    create_audio_config,
    create_transcription_result,
    create_tts_config,
    create_whisper_config,
    estimate_transcription_time,
    format_timestamp,
    get_audio_format,
    get_recommended_whisper_size,
    get_speech_task,
    get_whisper_size,
    list_audio_formats,
    list_speech_tasks,
    list_supported_languages,
    list_whisper_sizes,
    validate_audio_config,
    validate_whisper_config,
)


class TestWhisperSize:
    """Tests for WhisperSize enum."""

    def test_all_sizes_have_values(self) -> None:
        """All sizes have string values."""
        for size in WhisperSize:
            assert isinstance(size.value, str)

    def test_base_value(self) -> None:
        """Base has correct value."""
        assert WhisperSize.BASE.value == "base"

    def test_large_v3_value(self) -> None:
        """Large V3 has correct value."""
        assert WhisperSize.LARGE_V3.value == "large-v3"

    def test_valid_sizes_frozenset(self) -> None:
        """VALID_WHISPER_SIZES is a frozenset."""
        assert isinstance(VALID_WHISPER_SIZES, frozenset)


class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_all_formats_have_values(self) -> None:
        """All formats have string values."""
        for fmt in AudioFormat:
            assert isinstance(fmt.value, str)

    def test_mp3_value(self) -> None:
        """MP3 has correct value."""
        assert AudioFormat.MP3.value == "mp3"

    def test_wav_value(self) -> None:
        """WAV has correct value."""
        assert AudioFormat.WAV.value == "wav"


class TestSpeechTask:
    """Tests for SpeechTask enum."""

    def test_all_tasks_have_values(self) -> None:
        """All tasks have string values."""
        for task in SpeechTask:
            assert isinstance(task.value, str)

    def test_transcribe_value(self) -> None:
        """Transcribe has correct value."""
        assert SpeechTask.TRANSCRIBE.value == "transcribe"


class TestWhisperConfig:
    """Tests for WhisperConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Whisper config."""
        config = WhisperConfig(
            model_size=WhisperSize.BASE,
            language="en",
            task=SpeechTask.TRANSCRIBE,
            return_timestamps=True,
        )
        assert config.model_size == WhisperSize.BASE

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = WhisperConfig(WhisperSize.BASE, "en", SpeechTask.TRANSCRIBE, True)
        with pytest.raises(AttributeError):
            config.language = "fr"  # type: ignore[misc]


class TestValidateWhisperConfig:
    """Tests for validate_whisper_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = WhisperConfig(WhisperSize.BASE, "en", SpeechTask.TRANSCRIBE, True)
        validate_whisper_config(config)

    def test_invalid_language_raises(self) -> None:
        """Invalid language raises ValueError."""
        config = WhisperConfig(WhisperSize.BASE, "invalid", SpeechTask.TRANSCRIBE, True)
        with pytest.raises(ValueError, match="language must be one of"):
            validate_whisper_config(config)


class TestAudioConfig:
    """Tests for AudioConfig dataclass."""

    def test_create_config(self) -> None:
        """Create audio config."""
        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            format=AudioFormat.MP3,
            bit_depth=16,
        )
        assert config.sample_rate == 16000


class TestValidateAudioConfig:
    """Tests for validate_audio_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = AudioConfig(16000, 1, AudioFormat.MP3, 16)
        validate_audio_config(config)

    def test_invalid_sample_rate_raises(self) -> None:
        """Invalid sample rate raises ValueError."""
        config = AudioConfig(0, 1, AudioFormat.MP3, 16)
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            validate_audio_config(config)

    def test_invalid_channels_raises(self) -> None:
        """Invalid channels raises ValueError."""
        config = AudioConfig(16000, 0, AudioFormat.MP3, 16)
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            validate_audio_config(config)

    def test_invalid_bit_depth_raises(self) -> None:
        """Invalid bit depth raises ValueError."""
        config = AudioConfig(16000, 1, AudioFormat.MP3, 12)
        with pytest.raises(ValueError, match="bit_depth must be one of"):
            validate_audio_config(config)


class TestCreateWhisperConfig:
    """Tests for create_whisper_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_whisper_config()
        assert config.model_size == WhisperSize.BASE
        assert config.language == "en"

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_whisper_config(model_size="large-v3", language="fr")
        assert config.model_size == WhisperSize.LARGE_V3
        assert config.language == "fr"

    def test_invalid_size_raises(self) -> None:
        """Invalid size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            create_whisper_config(model_size="invalid")

    def test_invalid_language_raises(self) -> None:
        """Invalid language raises ValueError."""
        with pytest.raises(ValueError, match="language must be one of"):
            create_whisper_config(language="invalid")

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            create_whisper_config(task="invalid")


class TestCreateAudioConfig:
    """Tests for create_audio_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_audio_config()
        assert config.sample_rate == 16000
        assert config.channels == 1

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_audio_config(format="wav", sample_rate=44100)
        assert config.format == AudioFormat.WAV
        assert config.sample_rate == 44100

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="format must be one of"):
            create_audio_config(format="invalid")

    def test_invalid_sample_rate_raises(self) -> None:
        """Invalid sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            create_audio_config(sample_rate=0)


class TestCreateTTSConfig:
    """Tests for create_tts_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_tts_config()
        assert config.model_id == "microsoft/speecht5_tts"
        assert config.voice == "default"

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_tts_config(model_id="bark", voice="v2", speed=1.5)
        assert config.model_id == "bark"
        assert config.voice == "v2"
        assert config.speed == 1.5

    def test_empty_model_id_raises(self) -> None:
        """Empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            create_tts_config(model_id="")

    def test_zero_speed_raises(self) -> None:
        """Zero speed raises ValueError."""
        with pytest.raises(ValueError, match="speed must be positive"):
            create_tts_config(speed=0.0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            create_tts_config(sample_rate=-1)


class TestCreateTranscriptionResult:
    """Tests for create_transcription_result function."""

    def test_default_result(self) -> None:
        """Create default result."""
        result = create_transcription_result("Hello world")
        assert result.text == "Hello world"
        assert result.language == "en"

    def test_with_segments(self) -> None:
        """Create result with segments."""
        result = create_transcription_result(
            "Hello",
            segments=("Hello",),
            timestamps=((0.0, 1.0),),
        )
        assert len(result.segments) == 1
        assert result.segments[0] == "Hello"

    def test_empty_text_raises(self) -> None:
        """Empty text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            create_transcription_result("")

    def test_invalid_confidence_raises(self) -> None:
        """Invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between"):
            create_transcription_result("Hello", confidence=1.5)


class TestEstimateTranscriptionTime:
    """Tests for estimate_transcription_time function."""

    def test_basic_estimate(self) -> None:
        """Basic time estimate."""
        time = estimate_transcription_time(60.0, WhisperSize.BASE)
        assert time > 0

    def test_larger_model_slower(self) -> None:
        """Larger model is slower."""
        time_base = estimate_transcription_time(60.0, WhisperSize.BASE)
        time_large = estimate_transcription_time(60.0, WhisperSize.LARGE_V3)
        assert time_large > time_base

    def test_zero_duration_raises(self) -> None:
        """Zero duration raises ValueError."""
        with pytest.raises(ValueError, match="audio_duration must be positive"):
            estimate_transcription_time(0.0, WhisperSize.BASE)


class TestCalculateAudioDuration:
    """Tests for calculate_audio_duration function."""

    def test_basic_calculation(self) -> None:
        """Basic duration calculation."""
        duration = calculate_audio_duration(
            file_size_bytes=160000,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
        )
        assert duration > 0

    def test_zero_file_size_raises(self) -> None:
        """Zero file size raises ValueError."""
        with pytest.raises(ValueError, match="file_size_bytes must be positive"):
            calculate_audio_duration(0, 16000, 1, 16)

    def test_zero_sample_rate_raises(self) -> None:
        """Zero sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            calculate_audio_duration(160000, 0, 1, 16)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_basic_format(self) -> None:
        """Basic timestamp format."""
        ts = format_timestamp(3661.5)
        assert "01:01:01" in ts

    def test_zero_timestamp(self) -> None:
        """Zero timestamp."""
        ts = format_timestamp(0.0)
        assert "00:00:00" in ts

    def test_negative_timestamp_raises(self) -> None:
        """Negative timestamp raises ValueError."""
        with pytest.raises(ValueError, match="seconds cannot be negative"):
            format_timestamp(-1.0)


class TestListWhisperSizes:
    """Tests for list_whisper_sizes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        sizes = list_whisper_sizes()
        assert sizes == sorted(sizes)

    def test_contains_base(self) -> None:
        """Contains base."""
        sizes = list_whisper_sizes()
        assert "base" in sizes


class TestListAudioFormats:
    """Tests for list_audio_formats function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        formats = list_audio_formats()
        assert formats == sorted(formats)

    def test_contains_mp3(self) -> None:
        """Contains mp3."""
        formats = list_audio_formats()
        assert "mp3" in formats


class TestListSpeechTasks:
    """Tests for list_speech_tasks function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        tasks = list_speech_tasks()
        assert tasks == sorted(tasks)

    def test_contains_transcribe(self) -> None:
        """Contains transcribe."""
        tasks = list_speech_tasks()
        assert "transcribe" in tasks


class TestListSupportedLanguages:
    """Tests for list_supported_languages function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        languages = list_supported_languages()
        assert languages == sorted(languages)

    def test_contains_english(self) -> None:
        """Contains english."""
        languages = list_supported_languages()
        assert "en" in languages


class TestGetWhisperSize:
    """Tests for get_whisper_size function."""

    def test_get_base(self) -> None:
        """Get base size."""
        assert get_whisper_size("base") == WhisperSize.BASE

    def test_get_large_v3(self) -> None:
        """Get large-v3 size."""
        assert get_whisper_size("large-v3") == WhisperSize.LARGE_V3

    def test_invalid_size_raises(self) -> None:
        """Invalid size raises ValueError."""
        with pytest.raises(ValueError, match="size must be one of"):
            get_whisper_size("invalid")


class TestGetAudioFormat:
    """Tests for get_audio_format function."""

    def test_get_mp3(self) -> None:
        """Get mp3 format."""
        assert get_audio_format("mp3") == AudioFormat.MP3

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="format must be one of"):
            get_audio_format("invalid")


class TestGetSpeechTask:
    """Tests for get_speech_task function."""

    def test_get_transcribe(self) -> None:
        """Get transcribe task."""
        assert get_speech_task("transcribe") == SpeechTask.TRANSCRIBE

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_speech_task("invalid")


class TestGetRecommendedWhisperSize:
    """Tests for get_recommended_whisper_size function."""

    def test_short_audio_recommends_base(self) -> None:
        """Short audio recommends base model."""
        size = get_recommended_whisper_size(30.0)
        assert size == WhisperSize.BASE

    def test_accuracy_priority_recommends_large(self) -> None:
        """Accuracy priority recommends larger model."""
        size = get_recommended_whisper_size(60.0, accuracy_priority=True)
        assert size == WhisperSize.LARGE_V3

    def test_invalid_duration_raises(self) -> None:
        """Invalid duration raises ValueError."""
        with pytest.raises(ValueError, match="audio_duration must be positive"):
            get_recommended_whisper_size(0.0)


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_create_config(self) -> None:
        """Create TTS config."""
        config = TTSConfig(
            model_id="microsoft/speecht5_tts",
            voice="default",
            sample_rate=16000,
            speed=1.0,
        )
        assert config.speed == 1.0


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_create_result(self) -> None:
        """Create transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            segments=("Hello", "world"),
            timestamps=((0.0, 0.5), (0.5, 1.0)),
            confidence=0.95,
        )
        assert result.text == "Hello world"


class TestValidConstants:
    """Tests for validation constants."""

    def test_valid_languages_frozenset(self) -> None:
        """VALID_LANGUAGES is a frozenset."""
        assert isinstance(VALID_LANGUAGES, frozenset)

    def test_valid_tasks_frozenset(self) -> None:
        """VALID_TASKS is a frozenset."""
        assert isinstance(VALID_TASKS, frozenset)

    def test_valid_audio_formats_frozenset(self) -> None:
        """VALID_AUDIO_FORMATS is a frozenset."""
        assert isinstance(VALID_AUDIO_FORMATS, frozenset)
