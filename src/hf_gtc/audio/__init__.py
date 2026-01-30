"""Audio processing recipes for HuggingFace models.

This module provides utilities for speech-to-text, text-to-speech,
and audio processing.

Examples:
    >>> from hf_gtc.audio import WhisperConfig, AudioFormat
    >>> config = WhisperConfig(
    ...     model_size="base",
    ...     language="en",
    ...     task="transcribe",
    ...     return_timestamps=True,
    ... )
    >>> config.model_size
    'base'
"""

from __future__ import annotations

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

__all__: list[str] = [
    "VALID_AUDIO_FORMATS",
    "VALID_LANGUAGES",
    "VALID_TASKS",
    "VALID_WHISPER_SIZES",
    "AudioConfig",
    "AudioFormat",
    "SpeechTask",
    "TTSConfig",
    "TranscriptionResult",
    "WhisperConfig",
    "WhisperSize",
    "calculate_audio_duration",
    "create_audio_config",
    "create_transcription_result",
    "create_tts_config",
    "create_whisper_config",
    "estimate_transcription_time",
    "format_timestamp",
    "get_audio_format",
    "get_recommended_whisper_size",
    "get_speech_task",
    "get_whisper_size",
    "list_audio_formats",
    "list_speech_tasks",
    "list_supported_languages",
    "list_whisper_sizes",
    "validate_audio_config",
    "validate_whisper_config",
]
