"""Speech processing utilities.

This module provides functions for speech-to-text (Whisper) and
text-to-speech configurations.

Examples:
    >>> from hf_gtc.audio.speech import create_whisper_config
    >>> config = create_whisper_config(model_size="base", language="en")
    >>> config.model_size
    <WhisperSize.BASE: 'base'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class WhisperSize(Enum):
    """Whisper model sizes.

    Attributes:
        TINY: Tiny model (~39M params).
        BASE: Base model (~74M params).
        SMALL: Small model (~244M params).
        MEDIUM: Medium model (~769M params).
        LARGE: Large model (~1.5B params).
        LARGE_V2: Large v2 model.
        LARGE_V3: Large v3 model.

    Examples:
        >>> WhisperSize.BASE.value
        'base'
        >>> WhisperSize.LARGE_V3.value
        'large-v3'
    """

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


VALID_WHISPER_SIZES = frozenset(s.value for s in WhisperSize)


class SpeechTask(Enum):
    """Speech processing tasks.

    Attributes:
        TRANSCRIBE: Transcription (speech-to-text).
        TRANSLATE: Translation to English.

    Examples:
        >>> SpeechTask.TRANSCRIBE.value
        'transcribe'
        >>> SpeechTask.TRANSLATE.value
        'translate'
    """

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


VALID_TASKS = frozenset(t.value for t in SpeechTask)


class AudioFormat(Enum):
    """Supported audio formats.

    Attributes:
        WAV: WAV format.
        MP3: MP3 format.
        FLAC: FLAC format.
        OGG: OGG Vorbis format.
        M4A: M4A format.
        WEBM: WebM format.

    Examples:
        >>> AudioFormat.WAV.value
        'wav'
        >>> AudioFormat.MP3.value
        'mp3'
    """

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    WEBM = "webm"


VALID_AUDIO_FORMATS = frozenset(f.value for f in AudioFormat)

# Supported languages (subset of Whisper languages)
LanguageCode = Literal["en", "es", "fr", "de", "it", "pt", "nl", "ja", "zh", "ko", "ru"]
VALID_LANGUAGES = frozenset(
    {"en", "es", "fr", "de", "it", "pt", "nl", "ja", "zh", "ko", "ru"}
)


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    """Configuration for Whisper model.

    Attributes:
        model_size: Whisper model size.
        language: Target language code.
        task: Speech task (transcribe/translate).
        return_timestamps: Whether to return timestamps.

    Examples:
        >>> config = WhisperConfig(
        ...     model_size=WhisperSize.BASE,
        ...     language="en",
        ...     task=SpeechTask.TRANSCRIBE,
        ...     return_timestamps=True,
        ... )
        >>> config.model_size
        <WhisperSize.BASE: 'base'>
    """

    model_size: WhisperSize
    language: str
    task: SpeechTask
    return_timestamps: bool


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Configuration for audio processing.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        format: Audio format.
        bit_depth: Bit depth (8, 16, 24, 32).

    Examples:
        >>> config = AudioConfig(
        ...     sample_rate=16000,
        ...     channels=1,
        ...     format=AudioFormat.WAV,
        ...     bit_depth=16,
        ... )
        >>> config.sample_rate
        16000
    """

    sample_rate: int
    channels: int
    format: AudioFormat
    bit_depth: int


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Result from speech transcription.

    Attributes:
        text: Transcribed text.
        language: Detected language.
        segments: Tuple of segment texts.
        timestamps: Tuple of (start, end) timestamps.
        confidence: Confidence score (0-1).

    Examples:
        >>> result = TranscriptionResult(
        ...     text="Hello world",
        ...     language="en",
        ...     segments=("Hello", "world"),
        ...     timestamps=((0.0, 0.5), (0.5, 1.0)),
        ...     confidence=0.95,
        ... )
        >>> result.text
        'Hello world'
    """

    text: str
    language: str
    segments: tuple[str, ...]
    timestamps: tuple[tuple[float, float], ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class TTSConfig:
    """Configuration for text-to-speech.

    Attributes:
        model_id: TTS model identifier.
        voice: Voice/speaker identifier.
        sample_rate: Output sample rate.
        speed: Speech speed multiplier.

    Examples:
        >>> config = TTSConfig(
        ...     model_id="microsoft/speecht5_tts",
        ...     voice="default",
        ...     sample_rate=16000,
        ...     speed=1.0,
        ... )
        >>> config.speed
        1.0
    """

    model_id: str
    voice: str
    sample_rate: int
    speed: float


def validate_whisper_config(config: WhisperConfig) -> None:
    """Validate Whisper configuration.

    Args:
        config: Whisper configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = WhisperConfig(
        ...     WhisperSize.BASE, "en", SpeechTask.TRANSCRIBE, True
        ... )
        >>> validate_whisper_config(config)  # No error

        >>> bad = WhisperConfig(WhisperSize.BASE, "xx", SpeechTask.TRANSCRIBE, True)
        >>> validate_whisper_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: language must be one of
    """
    if config.language not in VALID_LANGUAGES:
        msg = f"language must be one of {VALID_LANGUAGES}, got '{config.language}'"
        raise ValueError(msg)


def validate_audio_config(config: AudioConfig) -> None:
    """Validate audio configuration.

    Args:
        config: Audio configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = AudioConfig(16000, 1, AudioFormat.WAV, 16)
        >>> validate_audio_config(config)  # No error

        >>> bad = AudioConfig(0, 1, AudioFormat.WAV, 16)
        >>> validate_audio_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sample_rate must be positive
    """
    if config.sample_rate <= 0:
        msg = f"sample_rate must be positive, got {config.sample_rate}"
        raise ValueError(msg)

    if config.channels not in (1, 2):
        msg = f"channels must be 1 or 2, got {config.channels}"
        raise ValueError(msg)

    valid_bit_depths = {8, 16, 24, 32}
    if config.bit_depth not in valid_bit_depths:
        msg = f"bit_depth must be one of {valid_bit_depths}, got {config.bit_depth}"
        raise ValueError(msg)


def create_whisper_config(
    model_size: str = "base",
    language: str = "en",
    task: str = "transcribe",
    return_timestamps: bool = True,
) -> WhisperConfig:
    """Create a Whisper configuration.

    Args:
        model_size: Whisper model size. Defaults to "base".
        language: Target language. Defaults to "en".
        task: Speech task. Defaults to "transcribe".
        return_timestamps: Return timestamps. Defaults to True.

    Returns:
        WhisperConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_whisper_config(model_size="small", language="es")
        >>> config.model_size
        <WhisperSize.SMALL: 'small'>

        >>> create_whisper_config(model_size="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size must be one of
    """
    if model_size not in VALID_WHISPER_SIZES:
        msg = f"model_size must be one of {VALID_WHISPER_SIZES}, got '{model_size}'"
        raise ValueError(msg)

    if task not in VALID_TASKS:
        msg = f"task must be one of {VALID_TASKS}, got '{task}'"
        raise ValueError(msg)

    config = WhisperConfig(
        model_size=WhisperSize(model_size),
        language=language,
        task=SpeechTask(task),
        return_timestamps=return_timestamps,
    )
    validate_whisper_config(config)
    return config


def create_audio_config(
    sample_rate: int = 16000,
    channels: int = 1,
    format: str = "wav",
    bit_depth: int = 16,
) -> AudioConfig:
    """Create an audio configuration.

    Args:
        sample_rate: Sample rate in Hz. Defaults to 16000.
        channels: Number of channels. Defaults to 1.
        format: Audio format. Defaults to "wav".
        bit_depth: Bit depth. Defaults to 16.

    Returns:
        AudioConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_audio_config(sample_rate=22050)
        >>> config.sample_rate
        22050

        >>> create_audio_config(format="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: format must be one of
    """
    if format not in VALID_AUDIO_FORMATS:
        msg = f"format must be one of {VALID_AUDIO_FORMATS}, got '{format}'"
        raise ValueError(msg)

    config = AudioConfig(
        sample_rate=sample_rate,
        channels=channels,
        format=AudioFormat(format),
        bit_depth=bit_depth,
    )
    validate_audio_config(config)
    return config


def create_tts_config(
    model_id: str = "microsoft/speecht5_tts",
    voice: str = "default",
    sample_rate: int = 16000,
    speed: float = 1.0,
) -> TTSConfig:
    """Create a TTS configuration.

    Args:
        model_id: TTS model ID. Defaults to "microsoft/speecht5_tts".
        voice: Voice identifier. Defaults to "default".
        sample_rate: Output sample rate. Defaults to 16000.
        speed: Speech speed. Defaults to 1.0.

    Returns:
        TTSConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_tts_config(speed=1.2)
        >>> config.speed
        1.2

        >>> create_tts_config(model_id="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_id cannot be empty
    """
    if not model_id:
        msg = "model_id cannot be empty"
        raise ValueError(msg)

    if sample_rate <= 0:
        msg = f"sample_rate must be positive, got {sample_rate}"
        raise ValueError(msg)

    if speed <= 0:
        msg = f"speed must be positive, got {speed}"
        raise ValueError(msg)

    return TTSConfig(
        model_id=model_id,
        voice=voice,
        sample_rate=sample_rate,
        speed=speed,
    )


def create_transcription_result(
    text: str,
    language: str = "en",
    segments: tuple[str, ...] = (),
    timestamps: tuple[tuple[float, float], ...] = (),
    confidence: float = 1.0,
) -> TranscriptionResult:
    """Create a transcription result.

    Args:
        text: Transcribed text.
        language: Detected language. Defaults to "en".
        segments: Segment texts. Defaults to ().
        timestamps: Segment timestamps. Defaults to ().
        confidence: Confidence score. Defaults to 1.0.

    Returns:
        TranscriptionResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_transcription_result("Hello world", confidence=0.95)
        >>> result.confidence
        0.95

        >>> create_transcription_result("", confidence=0.9)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be empty
    """
    if not text:
        msg = "text cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= confidence <= 1.0:
        msg = f"confidence must be between 0.0 and 1.0, got {confidence}"
        raise ValueError(msg)

    return TranscriptionResult(
        text=text,
        language=language,
        segments=segments,
        timestamps=timestamps,
        confidence=confidence,
    )


def list_whisper_sizes() -> list[str]:
    """List supported Whisper model sizes.

    Returns:
        Sorted list of size names.

    Examples:
        >>> sizes = list_whisper_sizes()
        >>> "base" in sizes
        True
        >>> "large-v3" in sizes
        True
    """
    return sorted(VALID_WHISPER_SIZES)


def list_speech_tasks() -> list[str]:
    """List supported speech tasks.

    Returns:
        Sorted list of task names.

    Examples:
        >>> tasks = list_speech_tasks()
        >>> "transcribe" in tasks
        True
        >>> "translate" in tasks
        True
    """
    return sorted(VALID_TASKS)


def list_audio_formats() -> list[str]:
    """List supported audio formats.

    Returns:
        Sorted list of format names.

    Examples:
        >>> formats = list_audio_formats()
        >>> "wav" in formats
        True
        >>> "mp3" in formats
        True
    """
    return sorted(VALID_AUDIO_FORMATS)


def list_supported_languages() -> list[str]:
    """List supported languages.

    Returns:
        Sorted list of language codes.

    Examples:
        >>> langs = list_supported_languages()
        >>> "en" in langs
        True
        >>> "es" in langs
        True
    """
    return sorted(VALID_LANGUAGES)


def get_whisper_size(name: str) -> WhisperSize:
    """Get Whisper size from name.

    Args:
        name: Size name.

    Returns:
        WhisperSize enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_whisper_size("base")
        <WhisperSize.BASE: 'base'>

        >>> get_whisper_size("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: size must be one of
    """
    if name not in VALID_WHISPER_SIZES:
        msg = f"size must be one of {VALID_WHISPER_SIZES}, got '{name}'"
        raise ValueError(msg)
    return WhisperSize(name)


def get_speech_task(name: str) -> SpeechTask:
    """Get speech task from name.

    Args:
        name: Task name.

    Returns:
        SpeechTask enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_speech_task("transcribe")
        <SpeechTask.TRANSCRIBE: 'transcribe'>

        >>> get_speech_task("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of
    """
    if name not in VALID_TASKS:
        msg = f"task must be one of {VALID_TASKS}, got '{name}'"
        raise ValueError(msg)
    return SpeechTask(name)


def get_audio_format(name: str) -> AudioFormat:
    """Get audio format from name.

    Args:
        name: Format name.

    Returns:
        AudioFormat enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_audio_format("wav")
        <AudioFormat.WAV: 'wav'>

        >>> get_audio_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: format must be one of
    """
    if name not in VALID_AUDIO_FORMATS:
        msg = f"format must be one of {VALID_AUDIO_FORMATS}, got '{name}'"
        raise ValueError(msg)
    return AudioFormat(name)


def get_recommended_whisper_size(
    audio_duration: float,
    accuracy_priority: bool = False,
) -> WhisperSize:
    """Get recommended Whisper size based on requirements.

    Args:
        audio_duration: Audio duration in seconds.
        accuracy_priority: Prioritize accuracy over speed.

    Returns:
        Recommended WhisperSize.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> get_recommended_whisper_size(60.0)
        <WhisperSize.BASE: 'base'>

        >>> get_recommended_whisper_size(300.0, accuracy_priority=True)
        <WhisperSize.LARGE_V3: 'large-v3'>
    """
    if audio_duration <= 0:
        msg = f"audio_duration must be positive, got {audio_duration}"
        raise ValueError(msg)

    if accuracy_priority:
        return WhisperSize.LARGE_V3

    # Recommend based on duration and typical use cases
    if audio_duration < 60:
        return WhisperSize.BASE
    elif audio_duration < 300:
        return WhisperSize.SMALL
    elif audio_duration < 1800:
        return WhisperSize.MEDIUM
    else:
        return WhisperSize.LARGE_V3


def estimate_transcription_time(
    audio_duration: float,
    model_size: WhisperSize,
    has_gpu: bool = True,
) -> float:
    """Estimate transcription time.

    Args:
        audio_duration: Audio duration in seconds.
        model_size: Whisper model size.
        has_gpu: Whether GPU is available.

    Returns:
        Estimated time in seconds.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> time = estimate_transcription_time(60.0, WhisperSize.BASE)
        >>> time > 0
        True

        >>> estimate_transcription_time(0, WhisperSize.BASE)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: audio_duration must be positive
    """
    if audio_duration <= 0:
        msg = f"audio_duration must be positive, got {audio_duration}"
        raise ValueError(msg)

    # Real-time factors (audio duration / processing time)
    rtf_gpu = {
        WhisperSize.TINY: 0.05,
        WhisperSize.BASE: 0.1,
        WhisperSize.SMALL: 0.2,
        WhisperSize.MEDIUM: 0.4,
        WhisperSize.LARGE: 0.8,
        WhisperSize.LARGE_V2: 0.8,
        WhisperSize.LARGE_V3: 0.9,
    }

    rtf = rtf_gpu[model_size]
    if not has_gpu:
        rtf *= 10  # CPU is ~10x slower

    return audio_duration * rtf


def calculate_audio_duration(
    file_size_bytes: int,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> float:
    """Calculate audio duration from file parameters.

    Args:
        file_size_bytes: File size in bytes.
        sample_rate: Sample rate in Hz.
        channels: Number of channels.
        bit_depth: Bit depth.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> duration = calculate_audio_duration(1920000, 16000, 1, 16)
        >>> abs(duration - 60.0) < 0.1
        True

        >>> calculate_audio_duration(0, 16000, 1, 16)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: file_size_bytes must be positive
    """
    if file_size_bytes <= 0:
        msg = f"file_size_bytes must be positive, got {file_size_bytes}"
        raise ValueError(msg)

    if sample_rate <= 0:
        msg = f"sample_rate must be positive, got {sample_rate}"
        raise ValueError(msg)

    if channels <= 0:
        msg = f"channels must be positive, got {channels}"
        raise ValueError(msg)

    if bit_depth <= 0:
        msg = f"bit_depth must be positive, got {bit_depth}"
        raise ValueError(msg)

    bytes_per_sample = bit_depth // 8
    bytes_per_second = sample_rate * channels * bytes_per_sample
    return file_size_bytes / bytes_per_second


def format_timestamp(seconds: float) -> str:
    """Format seconds as timestamp string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp (HH:MM:SS.mmm).

    Raises:
        ValueError: If seconds is negative.

    Examples:
        >>> format_timestamp(3661.5)
        '01:01:01.500'

        >>> format_timestamp(90.25)
        '00:01:30.250'

        >>> format_timestamp(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: seconds cannot be negative
    """
    if seconds < 0:
        msg = f"seconds cannot be negative, got {seconds}"
        raise ValueError(msg)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
