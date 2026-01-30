"""Music generation utilities.

This module provides functions for music generation configurations
including MusicGen, AudioLDM, and other music generation models.

Examples:
    >>> from hf_gtc.audio.music import create_music_gen_config
    >>> config = create_music_gen_config(model_type="musicgen", duration_seconds=30)
    >>> config.model_type
    <MusicModelType.MUSICGEN: 'musicgen'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class MusicModelType(Enum):
    """Music generation model types.

    Attributes:
        MUSICGEN: Meta MusicGen model.
        AUDIOLDM: AudioLDM model.
        AUDIOLDM2: AudioLDM 2 model.
        RIFFUSION: Riffusion model.
        MUSICLM: Google MusicLM model.

    Examples:
        >>> MusicModelType.MUSICGEN.value
        'musicgen'
        >>> MusicModelType.AUDIOLDM2.value
        'audioldm2'
    """

    MUSICGEN = "musicgen"
    AUDIOLDM = "audioldm"
    AUDIOLDM2 = "audioldm2"
    RIFFUSION = "riffusion"
    MUSICLM = "musiclm"


VALID_MUSIC_MODEL_TYPES = frozenset(m.value for m in MusicModelType)


class MusicGenre(Enum):
    """Music genres for generation.

    Attributes:
        POP: Pop music.
        ROCK: Rock music.
        CLASSICAL: Classical music.
        JAZZ: Jazz music.
        ELECTRONIC: Electronic music.
        AMBIENT: Ambient music.
        HIP_HOP: Hip-hop music.

    Examples:
        >>> MusicGenre.POP.value
        'pop'
        >>> MusicGenre.HIP_HOP.value
        'hip_hop'
    """

    POP = "pop"
    ROCK = "rock"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    AMBIENT = "ambient"
    HIP_HOP = "hip_hop"


VALID_MUSIC_GENRES = frozenset(g.value for g in MusicGenre)


class AudioQuality(Enum):
    """Audio quality levels.

    Attributes:
        LOW: Low quality (e.g., 64 kbps).
        MEDIUM: Medium quality (e.g., 128 kbps).
        HIGH: High quality (e.g., 320 kbps).
        LOSSLESS: Lossless quality.

    Examples:
        >>> AudioQuality.LOW.value
        'low'
        >>> AudioQuality.LOSSLESS.value
        'lossless'
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"


VALID_AUDIO_QUALITIES = frozenset(q.value for q in AudioQuality)

# Bitrate mapping for file size estimation (kbps)
QUALITY_BITRATES = {
    AudioQuality.LOW: 64,
    AudioQuality.MEDIUM: 128,
    AudioQuality.HIGH: 320,
    AudioQuality.LOSSLESS: 1411,  # CD quality 16-bit/44.1kHz stereo
}


@dataclass(frozen=True, slots=True)
class MusicGenConfig:
    """Configuration for music generation models.

    Attributes:
        model_type: Type of music generation model.
        duration_seconds: Target duration in seconds.
        sample_rate: Audio sample rate in Hz.
        guidance_scale: Classifier-free guidance scale.

    Examples:
        >>> config = MusicGenConfig(
        ...     model_type=MusicModelType.MUSICGEN,
        ...     duration_seconds=30,
        ...     sample_rate=32000,
        ...     guidance_scale=3.0,
        ... )
        >>> config.duration_seconds
        30
    """

    model_type: MusicModelType
    duration_seconds: int
    sample_rate: int
    guidance_scale: float


@dataclass(frozen=True, slots=True)
class MusicConditioningConfig:
    """Configuration for music conditioning inputs.

    Attributes:
        text_prompt: Text description for generation.
        melody_conditioning: Whether to use melody conditioning.
        style_conditioning: Whether to use style conditioning.

    Examples:
        >>> config = MusicConditioningConfig(
        ...     text_prompt="upbeat electronic dance music",
        ...     melody_conditioning=False,
        ...     style_conditioning=True,
        ... )
        >>> config.text_prompt
        'upbeat electronic dance music'
    """

    text_prompt: str
    melody_conditioning: bool
    style_conditioning: bool


@dataclass(frozen=True, slots=True)
class AudioOutputConfig:
    """Configuration for audio output.

    Attributes:
        format: Audio format (wav, mp3, flac, ogg).
        quality: Audio quality level.
        channels: Number of audio channels (1 or 2).
        normalize: Whether to normalize audio.

    Examples:
        >>> config = AudioOutputConfig(
        ...     format="wav",
        ...     quality=AudioQuality.HIGH,
        ...     channels=2,
        ...     normalize=True,
        ... )
        >>> config.format
        'wav'
    """

    format: str
    quality: AudioQuality
    channels: int
    normalize: bool


@dataclass(frozen=True, slots=True)
class MusicGenerationStats:
    """Statistics from music generation.

    Attributes:
        generation_time: Time to generate in seconds.
        duration_generated: Duration of generated audio in seconds.
        samples_per_second: Generation speed (samples/second).

    Examples:
        >>> stats = MusicGenerationStats(
        ...     generation_time=15.0,
        ...     duration_generated=30.0,
        ...     samples_per_second=64000.0,
        ... )
        >>> stats.generation_time
        15.0
    """

    generation_time: float
    duration_generated: float
    samples_per_second: float


@dataclass(frozen=True, slots=True)
class MelodyConfig:
    """Configuration for melody conditioning.

    Attributes:
        melody_audio_path: Path to melody reference audio.
        melody_influence: Influence strength (0.0-1.0).
        tempo_bpm: Target tempo in BPM.

    Examples:
        >>> config = MelodyConfig(
        ...     melody_audio_path="/path/to/melody.wav",
        ...     melody_influence=0.5,
        ...     tempo_bpm=120,
        ... )
        >>> config.tempo_bpm
        120
    """

    melody_audio_path: str
    melody_influence: float
    tempo_bpm: int


def validate_music_gen_config(config: MusicGenConfig) -> None:
    """Validate music generation configuration.

    Args:
        config: Music generation configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = MusicGenConfig(
        ...     MusicModelType.MUSICGEN, 30, 32000, 3.0
        ... )
        >>> validate_music_gen_config(config)  # No error

        >>> bad = MusicGenConfig(MusicModelType.MUSICGEN, 0, 32000, 3.0)
        >>> validate_music_gen_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: duration_seconds must be positive
    """
    if config.duration_seconds <= 0:
        msg = f"duration_seconds must be positive, got {config.duration_seconds}"
        raise ValueError(msg)

    if config.sample_rate <= 0:
        msg = f"sample_rate must be positive, got {config.sample_rate}"
        raise ValueError(msg)

    if config.guidance_scale < 0:
        msg = f"guidance_scale must be non-negative, got {config.guidance_scale}"
        raise ValueError(msg)


def validate_melody_config(config: MelodyConfig) -> None:
    """Validate melody configuration.

    Args:
        config: Melody configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = MelodyConfig("/path/to/melody.wav", 0.5, 120)
        >>> validate_melody_config(config)  # No error

        >>> bad = MelodyConfig("", 0.5, 120)
        >>> validate_melody_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: melody_audio_path cannot be empty

        >>> bad = MelodyConfig("/path/melody.wav", 1.5, 120)
        >>> validate_melody_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: melody_influence must be between 0.0 and 1.0
    """
    if not config.melody_audio_path:
        msg = "melody_audio_path cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= config.melody_influence <= 1.0:
        influence = config.melody_influence
        msg = f"melody_influence must be between 0.0 and 1.0, got {influence}"
        raise ValueError(msg)

    if config.tempo_bpm <= 0:
        msg = f"tempo_bpm must be positive, got {config.tempo_bpm}"
        raise ValueError(msg)


def create_music_gen_config(
    model_type: str = "musicgen",
    duration_seconds: int = 30,
    sample_rate: int = 32000,
    guidance_scale: float = 3.0,
) -> MusicGenConfig:
    """Create a music generation configuration.

    Args:
        model_type: Model type name. Defaults to "musicgen".
        duration_seconds: Target duration in seconds. Defaults to 30.
        sample_rate: Sample rate in Hz. Defaults to 32000.
        guidance_scale: Guidance scale. Defaults to 3.0.

    Returns:
        MusicGenConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_music_gen_config(duration_seconds=60)
        >>> config.duration_seconds
        60

        >>> create_music_gen_config(model_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of
    """
    if model_type not in VALID_MUSIC_MODEL_TYPES:
        msg = f"model_type must be one of {VALID_MUSIC_MODEL_TYPES}, got '{model_type}'"
        raise ValueError(msg)

    config = MusicGenConfig(
        model_type=MusicModelType(model_type),
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        guidance_scale=guidance_scale,
    )
    validate_music_gen_config(config)
    return config


def create_music_conditioning_config(
    text_prompt: str,
    melody_conditioning: bool = False,
    style_conditioning: bool = False,
) -> MusicConditioningConfig:
    """Create a music conditioning configuration.

    Args:
        text_prompt: Text description for generation.
        melody_conditioning: Use melody conditioning. Defaults to False.
        style_conditioning: Use style conditioning. Defaults to False.

    Returns:
        MusicConditioningConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_music_conditioning_config("upbeat jazz piano")
        >>> config.text_prompt
        'upbeat jazz piano'

        >>> create_music_conditioning_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text_prompt cannot be empty
    """
    if not text_prompt:
        msg = "text_prompt cannot be empty"
        raise ValueError(msg)

    return MusicConditioningConfig(
        text_prompt=text_prompt,
        melody_conditioning=melody_conditioning,
        style_conditioning=style_conditioning,
    )


def create_audio_output_config(
    format: str = "wav",
    quality: str = "high",
    channels: int = 2,
    normalize: bool = True,
) -> AudioOutputConfig:
    """Create an audio output configuration.

    Args:
        format: Audio format. Defaults to "wav".
        quality: Quality level. Defaults to "high".
        channels: Number of channels. Defaults to 2.
        normalize: Normalize audio. Defaults to True.

    Returns:
        AudioOutputConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_audio_output_config(format="mp3", channels=1)
        >>> config.format
        'mp3'

        >>> create_audio_output_config(quality="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quality must be one of

        >>> create_audio_output_config(channels=5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: channels must be 1 or 2
    """
    valid_formats = {"wav", "mp3", "flac", "ogg"}
    if format not in valid_formats:
        msg = f"format must be one of {valid_formats}, got '{format}'"
        raise ValueError(msg)

    if quality not in VALID_AUDIO_QUALITIES:
        msg = f"quality must be one of {VALID_AUDIO_QUALITIES}, got '{quality}'"
        raise ValueError(msg)

    if channels not in (1, 2):
        msg = f"channels must be 1 or 2, got {channels}"
        raise ValueError(msg)

    return AudioOutputConfig(
        format=format,
        quality=AudioQuality(quality),
        channels=channels,
        normalize=normalize,
    )


def list_music_model_types() -> list[str]:
    """List supported music model types.

    Returns:
        Sorted list of model type names.

    Examples:
        >>> types = list_music_model_types()
        >>> "musicgen" in types
        True
        >>> "audioldm" in types
        True
    """
    return sorted(VALID_MUSIC_MODEL_TYPES)


def list_music_genres() -> list[str]:
    """List supported music genres.

    Returns:
        Sorted list of genre names.

    Examples:
        >>> genres = list_music_genres()
        >>> "pop" in genres
        True
        >>> "jazz" in genres
        True
    """
    return sorted(VALID_MUSIC_GENRES)


def list_audio_qualities() -> list[str]:
    """List supported audio quality levels.

    Returns:
        Sorted list of quality level names.

    Examples:
        >>> qualities = list_audio_qualities()
        >>> "high" in qualities
        True
        >>> "lossless" in qualities
        True
    """
    return sorted(VALID_AUDIO_QUALITIES)


def get_music_model_type(name: str) -> MusicModelType:
    """Get music model type from name.

    Args:
        name: Model type name.

    Returns:
        MusicModelType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_music_model_type("musicgen")
        <MusicModelType.MUSICGEN: 'musicgen'>

        >>> get_music_model_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of
    """
    if name not in VALID_MUSIC_MODEL_TYPES:
        msg = f"model_type must be one of {VALID_MUSIC_MODEL_TYPES}, got '{name}'"
        raise ValueError(msg)
    return MusicModelType(name)


def get_music_genre(name: str) -> MusicGenre:
    """Get music genre from name.

    Args:
        name: Genre name.

    Returns:
        MusicGenre enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_music_genre("jazz")
        <MusicGenre.JAZZ: 'jazz'>

        >>> get_music_genre("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: genre must be one of
    """
    if name not in VALID_MUSIC_GENRES:
        msg = f"genre must be one of {VALID_MUSIC_GENRES}, got '{name}'"
        raise ValueError(msg)
    return MusicGenre(name)


def get_audio_quality(name: str) -> AudioQuality:
    """Get audio quality from name.

    Args:
        name: Quality level name.

    Returns:
        AudioQuality enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_audio_quality("high")
        <AudioQuality.HIGH: 'high'>

        >>> get_audio_quality("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quality must be one of
    """
    if name not in VALID_AUDIO_QUALITIES:
        msg = f"quality must be one of {VALID_AUDIO_QUALITIES}, got '{name}'"
        raise ValueError(msg)
    return AudioQuality(name)


def estimate_generation_time(
    duration_seconds: int,
    model_type: MusicModelType,
    has_gpu: bool = True,
) -> float:
    """Estimate time to generate audio.

    Args:
        duration_seconds: Target audio duration in seconds.
        model_type: Music generation model type.
        has_gpu: Whether GPU is available.

    Returns:
        Estimated generation time in seconds.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> time = estimate_generation_time(30, MusicModelType.MUSICGEN)
        >>> time > 0
        True

        >>> estimate_generation_time(0, MusicModelType.MUSICGEN)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: duration_seconds must be positive
    """
    if duration_seconds <= 0:
        msg = f"duration_seconds must be positive, got {duration_seconds}"
        raise ValueError(msg)

    # Real-time factors (generation time / audio duration)
    # These are approximate based on typical GPU performance
    rtf_gpu = {
        MusicModelType.MUSICGEN: 0.5,
        MusicModelType.AUDIOLDM: 0.8,
        MusicModelType.AUDIOLDM2: 1.0,
        MusicModelType.RIFFUSION: 0.3,
        MusicModelType.MUSICLM: 1.2,
    }

    rtf = rtf_gpu[model_type]
    if not has_gpu:
        rtf *= 15  # CPU is ~15x slower for music generation

    return duration_seconds * rtf


def calculate_audio_file_size(
    duration_seconds: int,
    quality: AudioQuality,
    channels: int = 2,
) -> int:
    """Calculate estimated audio file size in bytes.

    Args:
        duration_seconds: Audio duration in seconds.
        quality: Audio quality level.
        channels: Number of audio channels.

    Returns:
        Estimated file size in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> size = calculate_audio_file_size(60, AudioQuality.HIGH)
        >>> size > 0
        True

        >>> calculate_audio_file_size(0, AudioQuality.HIGH)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: duration_seconds must be positive

        >>> calculate_audio_file_size(60, AudioQuality.LOW, channels=5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: channels must be 1 or 2
    """
    if duration_seconds <= 0:
        msg = f"duration_seconds must be positive, got {duration_seconds}"
        raise ValueError(msg)

    if channels not in (1, 2):
        msg = f"channels must be 1 or 2, got {channels}"
        raise ValueError(msg)

    # Get bitrate in kbps
    bitrate_kbps = QUALITY_BITRATES[quality]

    # Adjust for mono (stereo bitrates are for 2 channels)
    if channels == 1 and quality != AudioQuality.LOSSLESS:
        bitrate_kbps = bitrate_kbps // 2

    # Calculate size: bitrate (kbps) * duration (s) / 8 (bits to bytes) * 1000 (k)
    # Simplified: bitrate_kbps * duration * 125
    size_bytes = bitrate_kbps * duration_seconds * 125

    return size_bytes
