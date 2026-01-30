"""Tests for audio.music module."""

from __future__ import annotations

import pytest

from hf_gtc.audio.music import (
    QUALITY_BITRATES,
    VALID_AUDIO_QUALITIES,
    VALID_MUSIC_GENRES,
    VALID_MUSIC_MODEL_TYPES,
    AudioOutputConfig,
    AudioQuality,
    MelodyConfig,
    MusicConditioningConfig,
    MusicGenConfig,
    MusicGenerationStats,
    MusicGenre,
    MusicModelType,
    calculate_audio_file_size,
    create_audio_output_config,
    create_music_conditioning_config,
    create_music_gen_config,
    estimate_generation_time,
    get_audio_quality,
    get_music_genre,
    get_music_model_type,
    list_audio_qualities,
    list_music_genres,
    list_music_model_types,
    validate_melody_config,
    validate_music_gen_config,
)


class TestMusicModelType:
    """Tests for MusicModelType enum."""

    def test_all_types_have_values(self) -> None:
        """All model types have string values."""
        for model_type in MusicModelType:
            assert isinstance(model_type.value, str)

    def test_musicgen_value(self) -> None:
        """MUSICGEN has correct value."""
        assert MusicModelType.MUSICGEN.value == "musicgen"

    def test_audioldm_value(self) -> None:
        """AUDIOLDM has correct value."""
        assert MusicModelType.AUDIOLDM.value == "audioldm"

    def test_audioldm2_value(self) -> None:
        """AUDIOLDM2 has correct value."""
        assert MusicModelType.AUDIOLDM2.value == "audioldm2"

    def test_riffusion_value(self) -> None:
        """RIFFUSION has correct value."""
        assert MusicModelType.RIFFUSION.value == "riffusion"

    def test_musiclm_value(self) -> None:
        """MUSICLM has correct value."""
        assert MusicModelType.MUSICLM.value == "musiclm"

    def test_valid_model_types_frozenset(self) -> None:
        """VALID_MUSIC_MODEL_TYPES is a frozenset."""
        assert isinstance(VALID_MUSIC_MODEL_TYPES, frozenset)

    def test_valid_model_types_contains_all_enums(self) -> None:
        """VALID_MUSIC_MODEL_TYPES contains all enum values."""
        for model_type in MusicModelType:
            assert model_type.value in VALID_MUSIC_MODEL_TYPES


class TestMusicGenre:
    """Tests for MusicGenre enum."""

    def test_all_genres_have_values(self) -> None:
        """All genres have string values."""
        for genre in MusicGenre:
            assert isinstance(genre.value, str)

    def test_pop_value(self) -> None:
        """POP has correct value."""
        assert MusicGenre.POP.value == "pop"

    def test_rock_value(self) -> None:
        """ROCK has correct value."""
        assert MusicGenre.ROCK.value == "rock"

    def test_classical_value(self) -> None:
        """CLASSICAL has correct value."""
        assert MusicGenre.CLASSICAL.value == "classical"

    def test_jazz_value(self) -> None:
        """JAZZ has correct value."""
        assert MusicGenre.JAZZ.value == "jazz"

    def test_electronic_value(self) -> None:
        """ELECTRONIC has correct value."""
        assert MusicGenre.ELECTRONIC.value == "electronic"

    def test_ambient_value(self) -> None:
        """AMBIENT has correct value."""
        assert MusicGenre.AMBIENT.value == "ambient"

    def test_hip_hop_value(self) -> None:
        """HIP_HOP has correct value."""
        assert MusicGenre.HIP_HOP.value == "hip_hop"

    def test_valid_genres_frozenset(self) -> None:
        """VALID_MUSIC_GENRES is a frozenset."""
        assert isinstance(VALID_MUSIC_GENRES, frozenset)

    def test_valid_genres_contains_all_enums(self) -> None:
        """VALID_MUSIC_GENRES contains all enum values."""
        for genre in MusicGenre:
            assert genre.value in VALID_MUSIC_GENRES


class TestAudioQuality:
    """Tests for AudioQuality enum."""

    def test_all_qualities_have_values(self) -> None:
        """All qualities have string values."""
        for quality in AudioQuality:
            assert isinstance(quality.value, str)

    def test_low_value(self) -> None:
        """LOW has correct value."""
        assert AudioQuality.LOW.value == "low"

    def test_medium_value(self) -> None:
        """MEDIUM has correct value."""
        assert AudioQuality.MEDIUM.value == "medium"

    def test_high_value(self) -> None:
        """HIGH has correct value."""
        assert AudioQuality.HIGH.value == "high"

    def test_lossless_value(self) -> None:
        """LOSSLESS has correct value."""
        assert AudioQuality.LOSSLESS.value == "lossless"

    def test_valid_qualities_frozenset(self) -> None:
        """VALID_AUDIO_QUALITIES is a frozenset."""
        assert isinstance(VALID_AUDIO_QUALITIES, frozenset)

    def test_valid_qualities_contains_all_enums(self) -> None:
        """VALID_AUDIO_QUALITIES contains all enum values."""
        for quality in AudioQuality:
            assert quality.value in VALID_AUDIO_QUALITIES


class TestQualityBitrates:
    """Tests for QUALITY_BITRATES mapping."""

    def test_is_dict(self) -> None:
        """QUALITY_BITRATES is a dict."""
        assert isinstance(QUALITY_BITRATES, dict)

    def test_contains_all_qualities(self) -> None:
        """QUALITY_BITRATES contains all AudioQuality values."""
        for quality in AudioQuality:
            assert quality in QUALITY_BITRATES

    def test_low_bitrate(self) -> None:
        """LOW has expected bitrate."""
        assert QUALITY_BITRATES[AudioQuality.LOW] == 64

    def test_medium_bitrate(self) -> None:
        """MEDIUM has expected bitrate."""
        assert QUALITY_BITRATES[AudioQuality.MEDIUM] == 128

    def test_high_bitrate(self) -> None:
        """HIGH has expected bitrate."""
        assert QUALITY_BITRATES[AudioQuality.HIGH] == 320

    def test_lossless_bitrate(self) -> None:
        """LOSSLESS has expected bitrate."""
        assert QUALITY_BITRATES[AudioQuality.LOSSLESS] == 1411


class TestMusicGenConfig:
    """Tests for MusicGenConfig dataclass."""

    def test_create_config(self) -> None:
        """Create MusicGenConfig."""
        config = MusicGenConfig(
            model_type=MusicModelType.MUSICGEN,
            duration_seconds=30,
            sample_rate=32000,
            guidance_scale=3.0,
        )
        assert config.model_type == MusicModelType.MUSICGEN
        assert config.duration_seconds == 30
        assert config.sample_rate == 32000
        assert config.guidance_scale == 3.0

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 30, 32000, 3.0
        )
        with pytest.raises(AttributeError):
            config.duration_seconds = 60  # type: ignore[misc]

    def test_config_has_slots(self) -> None:
        """Config uses slots for memory efficiency."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 30, 32000, 3.0
        )
        assert hasattr(config, "__slots__") or not hasattr(config, "__dict__")


class TestMusicConditioningConfig:
    """Tests for MusicConditioningConfig dataclass."""

    def test_create_config(self) -> None:
        """Create MusicConditioningConfig."""
        config = MusicConditioningConfig(
            text_prompt="upbeat electronic dance music",
            melody_conditioning=False,
            style_conditioning=True,
        )
        assert config.text_prompt == "upbeat electronic dance music"
        assert config.melody_conditioning is False
        assert config.style_conditioning is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = MusicConditioningConfig(
            "upbeat jazz", False, True
        )
        with pytest.raises(AttributeError):
            config.text_prompt = "changed"  # type: ignore[misc]


class TestAudioOutputConfig:
    """Tests for AudioOutputConfig dataclass."""

    def test_create_config(self) -> None:
        """Create AudioOutputConfig."""
        config = AudioOutputConfig(
            format="wav",
            quality=AudioQuality.HIGH,
            channels=2,
            normalize=True,
        )
        assert config.format == "wav"
        assert config.quality == AudioQuality.HIGH
        assert config.channels == 2
        assert config.normalize is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = AudioOutputConfig(
            "mp3", AudioQuality.MEDIUM, 1, False
        )
        with pytest.raises(AttributeError):
            config.channels = 2  # type: ignore[misc]


class TestMusicGenerationStats:
    """Tests for MusicGenerationStats dataclass."""

    def test_create_stats(self) -> None:
        """Create MusicGenerationStats."""
        stats = MusicGenerationStats(
            generation_time=15.0,
            duration_generated=30.0,
            samples_per_second=64000.0,
        )
        assert stats.generation_time == 15.0
        assert stats.duration_generated == 30.0
        assert stats.samples_per_second == 64000.0

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = MusicGenerationStats(15.0, 30.0, 64000.0)
        with pytest.raises(AttributeError):
            stats.generation_time = 20.0  # type: ignore[misc]


class TestMelodyConfig:
    """Tests for MelodyConfig dataclass."""

    def test_create_config(self) -> None:
        """Create MelodyConfig."""
        config = MelodyConfig(
            melody_audio_path="/path/to/melody.wav",
            melody_influence=0.5,
            tempo_bpm=120,
        )
        assert config.melody_audio_path == "/path/to/melody.wav"
        assert config.melody_influence == 0.5
        assert config.tempo_bpm == 120

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = MelodyConfig("/path/melody.wav", 0.5, 120)
        with pytest.raises(AttributeError):
            config.tempo_bpm = 140  # type: ignore[misc]


class TestValidateMusicGenConfig:
    """Tests for validate_music_gen_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 30, 32000, 3.0
        )
        validate_music_gen_config(config)  # Should not raise

    def test_zero_duration_raises(self) -> None:
        """Zero duration raises ValueError."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 0, 32000, 3.0
        )
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            validate_music_gen_config(config)

    def test_negative_duration_raises(self) -> None:
        """Negative duration raises ValueError."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, -10, 32000, 3.0
        )
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            validate_music_gen_config(config)

    def test_zero_sample_rate_raises(self) -> None:
        """Zero sample rate raises ValueError."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 30, 0, 3.0
        )
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            validate_music_gen_config(config)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample rate raises ValueError."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 30, -16000, 3.0
        )
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            validate_music_gen_config(config)

    def test_negative_guidance_scale_raises(self) -> None:
        """Negative guidance scale raises ValueError."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 30, 32000, -1.0
        )
        with pytest.raises(ValueError, match="guidance_scale must be non-negative"):
            validate_music_gen_config(config)

    def test_zero_guidance_scale_valid(self) -> None:
        """Zero guidance scale is valid."""
        config = MusicGenConfig(
            MusicModelType.MUSICGEN, 30, 32000, 0.0
        )
        validate_music_gen_config(config)  # Should not raise


class TestValidateMelodyConfig:
    """Tests for validate_melody_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = MelodyConfig("/path/to/melody.wav", 0.5, 120)
        validate_melody_config(config)  # Should not raise

    def test_empty_path_raises(self) -> None:
        """Empty melody_audio_path raises ValueError."""
        config = MelodyConfig("", 0.5, 120)
        with pytest.raises(ValueError, match="melody_audio_path cannot be empty"):
            validate_melody_config(config)

    def test_influence_below_zero_raises(self) -> None:
        """Influence below 0 raises ValueError."""
        config = MelodyConfig("/path/melody.wav", -0.1, 120)
        with pytest.raises(ValueError, match="melody_influence must be between"):
            validate_melody_config(config)

    def test_influence_above_one_raises(self) -> None:
        """Influence above 1 raises ValueError."""
        config = MelodyConfig("/path/melody.wav", 1.5, 120)
        with pytest.raises(ValueError, match="melody_influence must be between"):
            validate_melody_config(config)

    def test_influence_boundary_zero_valid(self) -> None:
        """Influence of 0.0 is valid."""
        config = MelodyConfig("/path/melody.wav", 0.0, 120)
        validate_melody_config(config)  # Should not raise

    def test_influence_boundary_one_valid(self) -> None:
        """Influence of 1.0 is valid."""
        config = MelodyConfig("/path/melody.wav", 1.0, 120)
        validate_melody_config(config)  # Should not raise

    def test_zero_tempo_raises(self) -> None:
        """Zero tempo_bpm raises ValueError."""
        config = MelodyConfig("/path/melody.wav", 0.5, 0)
        with pytest.raises(ValueError, match="tempo_bpm must be positive"):
            validate_melody_config(config)

    def test_negative_tempo_raises(self) -> None:
        """Negative tempo_bpm raises ValueError."""
        config = MelodyConfig("/path/melody.wav", 0.5, -60)
        with pytest.raises(ValueError, match="tempo_bpm must be positive"):
            validate_melody_config(config)


class TestCreateMusicGenConfig:
    """Tests for create_music_gen_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_music_gen_config()
        assert config.model_type == MusicModelType.MUSICGEN
        assert config.duration_seconds == 30
        assert config.sample_rate == 32000
        assert config.guidance_scale == 3.0

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_music_gen_config(
            model_type="audioldm2",
            duration_seconds=60,
            sample_rate=44100,
            guidance_scale=5.0,
        )
        assert config.model_type == MusicModelType.AUDIOLDM2
        assert config.duration_seconds == 60
        assert config.sample_rate == 44100
        assert config.guidance_scale == 5.0

    @pytest.mark.parametrize("model_type", [
        "musicgen",
        "audioldm",
        "audioldm2",
        "riffusion",
        "musiclm",
    ])
    def test_all_valid_model_types(self, model_type: str) -> None:
        """All valid model types are accepted."""
        config = create_music_gen_config(model_type=model_type)
        assert config.model_type.value == model_type

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_music_gen_config(model_type="invalid")

    def test_zero_duration_raises(self) -> None:
        """Zero duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            create_music_gen_config(duration_seconds=0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            create_music_gen_config(sample_rate=-1)

    def test_negative_guidance_scale_raises(self) -> None:
        """Negative guidance scale raises ValueError."""
        with pytest.raises(ValueError, match="guidance_scale must be non-negative"):
            create_music_gen_config(guidance_scale=-1.0)


class TestCreateMusicConditioningConfig:
    """Tests for create_music_conditioning_config function."""

    def test_default_config(self) -> None:
        """Create default config with only required prompt."""
        config = create_music_conditioning_config("upbeat jazz piano")
        assert config.text_prompt == "upbeat jazz piano"
        assert config.melody_conditioning is False
        assert config.style_conditioning is False

    def test_custom_config(self) -> None:
        """Create custom config with all options."""
        config = create_music_conditioning_config(
            "ambient electronic",
            melody_conditioning=True,
            style_conditioning=True,
        )
        assert config.text_prompt == "ambient electronic"
        assert config.melody_conditioning is True
        assert config.style_conditioning is True

    def test_empty_prompt_raises(self) -> None:
        """Empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="text_prompt cannot be empty"):
            create_music_conditioning_config("")


class TestCreateAudioOutputConfig:
    """Tests for create_audio_output_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_audio_output_config()
        assert config.format == "wav"
        assert config.quality == AudioQuality.HIGH
        assert config.channels == 2
        assert config.normalize is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_audio_output_config(
            format="mp3",
            quality="low",
            channels=1,
            normalize=False,
        )
        assert config.format == "mp3"
        assert config.quality == AudioQuality.LOW
        assert config.channels == 1
        assert config.normalize is False

    @pytest.mark.parametrize("fmt", ["wav", "mp3", "flac", "ogg"])
    def test_all_valid_formats(self, fmt: str) -> None:
        """All valid formats are accepted."""
        config = create_audio_output_config(format=fmt)
        assert config.format == fmt

    @pytest.mark.parametrize("quality", ["low", "medium", "high", "lossless"])
    def test_all_valid_qualities(self, quality: str) -> None:
        """All valid qualities are accepted."""
        config = create_audio_output_config(quality=quality)
        assert config.quality.value == quality

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="format must be one of"):
            create_audio_output_config(format="invalid")

    def test_invalid_quality_raises(self) -> None:
        """Invalid quality raises ValueError."""
        with pytest.raises(ValueError, match="quality must be one of"):
            create_audio_output_config(quality="invalid")

    @pytest.mark.parametrize("channels", [1, 2])
    def test_valid_channels(self, channels: int) -> None:
        """Valid channel counts are accepted."""
        config = create_audio_output_config(channels=channels)
        assert config.channels == channels

    def test_invalid_channels_raises(self) -> None:
        """Invalid channel count raises ValueError."""
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            create_audio_output_config(channels=5)

    def test_zero_channels_raises(self) -> None:
        """Zero channels raises ValueError."""
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            create_audio_output_config(channels=0)


class TestListMusicModelTypes:
    """Tests for list_music_model_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_music_model_types()
        assert types == sorted(types)

    def test_contains_musicgen(self) -> None:
        """Contains musicgen."""
        types = list_music_model_types()
        assert "musicgen" in types

    def test_contains_audioldm(self) -> None:
        """Contains audioldm."""
        types = list_music_model_types()
        assert "audioldm" in types

    def test_returns_list(self) -> None:
        """Returns a list type."""
        types = list_music_model_types()
        assert isinstance(types, list)


class TestListMusicGenres:
    """Tests for list_music_genres function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        genres = list_music_genres()
        assert genres == sorted(genres)

    def test_contains_pop(self) -> None:
        """Contains pop."""
        genres = list_music_genres()
        assert "pop" in genres

    def test_contains_jazz(self) -> None:
        """Contains jazz."""
        genres = list_music_genres()
        assert "jazz" in genres

    def test_returns_list(self) -> None:
        """Returns a list type."""
        genres = list_music_genres()
        assert isinstance(genres, list)


class TestListAudioQualities:
    """Tests for list_audio_qualities function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        qualities = list_audio_qualities()
        assert qualities == sorted(qualities)

    def test_contains_high(self) -> None:
        """Contains high."""
        qualities = list_audio_qualities()
        assert "high" in qualities

    def test_contains_lossless(self) -> None:
        """Contains lossless."""
        qualities = list_audio_qualities()
        assert "lossless" in qualities

    def test_returns_list(self) -> None:
        """Returns a list type."""
        qualities = list_audio_qualities()
        assert isinstance(qualities, list)


class TestGetMusicModelType:
    """Tests for get_music_model_type function."""

    def test_get_musicgen(self) -> None:
        """Get musicgen model type."""
        assert get_music_model_type("musicgen") == MusicModelType.MUSICGEN

    def test_get_audioldm(self) -> None:
        """Get audioldm model type."""
        assert get_music_model_type("audioldm") == MusicModelType.AUDIOLDM

    def test_get_audioldm2(self) -> None:
        """Get audioldm2 model type."""
        assert get_music_model_type("audioldm2") == MusicModelType.AUDIOLDM2

    def test_get_riffusion(self) -> None:
        """Get riffusion model type."""
        assert get_music_model_type("riffusion") == MusicModelType.RIFFUSION

    def test_get_musiclm(self) -> None:
        """Get musiclm model type."""
        assert get_music_model_type("musiclm") == MusicModelType.MUSICLM

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_music_model_type("invalid")


class TestGetMusicGenre:
    """Tests for get_music_genre function."""

    def test_get_pop(self) -> None:
        """Get pop genre."""
        assert get_music_genre("pop") == MusicGenre.POP

    def test_get_jazz(self) -> None:
        """Get jazz genre."""
        assert get_music_genre("jazz") == MusicGenre.JAZZ

    def test_get_hip_hop(self) -> None:
        """Get hip_hop genre."""
        assert get_music_genre("hip_hop") == MusicGenre.HIP_HOP

    def test_invalid_genre_raises(self) -> None:
        """Invalid genre raises ValueError."""
        with pytest.raises(ValueError, match="genre must be one of"):
            get_music_genre("invalid")


class TestGetAudioQuality:
    """Tests for get_audio_quality function."""

    def test_get_low(self) -> None:
        """Get low quality."""
        assert get_audio_quality("low") == AudioQuality.LOW

    def test_get_high(self) -> None:
        """Get high quality."""
        assert get_audio_quality("high") == AudioQuality.HIGH

    def test_get_lossless(self) -> None:
        """Get lossless quality."""
        assert get_audio_quality("lossless") == AudioQuality.LOSSLESS

    def test_invalid_quality_raises(self) -> None:
        """Invalid quality raises ValueError."""
        with pytest.raises(ValueError, match="quality must be one of"):
            get_audio_quality("invalid")


class TestEstimateGenerationTime:
    """Tests for estimate_generation_time function."""

    def test_basic_estimate(self) -> None:
        """Basic time estimate."""
        time = estimate_generation_time(30, MusicModelType.MUSICGEN)
        assert time > 0

    def test_longer_duration_more_time(self) -> None:
        """Longer duration takes more time."""
        time_30 = estimate_generation_time(30, MusicModelType.MUSICGEN)
        time_60 = estimate_generation_time(60, MusicModelType.MUSICGEN)
        assert time_60 > time_30

    def test_no_gpu_slower(self) -> None:
        """Without GPU is slower."""
        time_gpu = estimate_generation_time(30, MusicModelType.MUSICGEN, has_gpu=True)
        time_cpu = estimate_generation_time(30, MusicModelType.MUSICGEN, has_gpu=False)
        assert time_cpu > time_gpu

    @pytest.mark.parametrize("model_type", list(MusicModelType))
    def test_all_model_types(self, model_type: MusicModelType) -> None:
        """All model types return positive time."""
        time = estimate_generation_time(30, model_type)
        assert time > 0

    def test_zero_duration_raises(self) -> None:
        """Zero duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            estimate_generation_time(0, MusicModelType.MUSICGEN)

    def test_negative_duration_raises(self) -> None:
        """Negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            estimate_generation_time(-10, MusicModelType.MUSICGEN)

    def test_riffusion_fastest(self) -> None:
        """Riffusion is the fastest model."""
        time_riffusion = estimate_generation_time(30, MusicModelType.RIFFUSION)
        time_musiclm = estimate_generation_time(30, MusicModelType.MUSICLM)
        assert time_riffusion < time_musiclm

    def test_cpu_multiplier_is_15x(self) -> None:
        """CPU is 15x slower than GPU."""
        time_gpu = estimate_generation_time(30, MusicModelType.MUSICGEN, has_gpu=True)
        time_cpu = estimate_generation_time(30, MusicModelType.MUSICGEN, has_gpu=False)
        assert time_cpu == time_gpu * 15


class TestCalculateAudioFileSize:
    """Tests for calculate_audio_file_size function."""

    def test_basic_calculation(self) -> None:
        """Basic file size calculation."""
        size = calculate_audio_file_size(60, AudioQuality.HIGH)
        assert size > 0

    def test_longer_duration_larger_file(self) -> None:
        """Longer duration produces larger file."""
        size_30 = calculate_audio_file_size(30, AudioQuality.HIGH)
        size_60 = calculate_audio_file_size(60, AudioQuality.HIGH)
        assert size_60 > size_30

    def test_higher_quality_larger_file(self) -> None:
        """Higher quality produces larger file."""
        size_low = calculate_audio_file_size(60, AudioQuality.LOW)
        size_high = calculate_audio_file_size(60, AudioQuality.HIGH)
        assert size_high > size_low

    def test_lossless_largest(self) -> None:
        """Lossless produces largest file."""
        size_high = calculate_audio_file_size(60, AudioQuality.HIGH)
        size_lossless = calculate_audio_file_size(60, AudioQuality.LOSSLESS)
        assert size_lossless > size_high

    def test_mono_smaller_than_stereo(self) -> None:
        """Mono is smaller than stereo."""
        size_mono = calculate_audio_file_size(60, AudioQuality.HIGH, channels=1)
        size_stereo = calculate_audio_file_size(60, AudioQuality.HIGH, channels=2)
        assert size_mono < size_stereo

    def test_zero_duration_raises(self) -> None:
        """Zero duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            calculate_audio_file_size(0, AudioQuality.HIGH)

    def test_negative_duration_raises(self) -> None:
        """Negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            calculate_audio_file_size(-10, AudioQuality.HIGH)

    def test_invalid_channels_raises(self) -> None:
        """Invalid channels raises ValueError."""
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            calculate_audio_file_size(60, AudioQuality.HIGH, channels=5)

    def test_zero_channels_raises(self) -> None:
        """Zero channels raises ValueError."""
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            calculate_audio_file_size(60, AudioQuality.HIGH, channels=0)

    @pytest.mark.parametrize("quality", list(AudioQuality))
    def test_all_qualities(self, quality: AudioQuality) -> None:
        """All qualities produce positive size."""
        size = calculate_audio_file_size(60, quality)
        assert size > 0

    def test_lossless_mono_same_as_stereo(self) -> None:
        """Lossless mono is same as stereo (no compression halving)."""
        size_mono = calculate_audio_file_size(60, AudioQuality.LOSSLESS, channels=1)
        size_stereo = calculate_audio_file_size(60, AudioQuality.LOSSLESS, channels=2)
        # For lossless, mono doesn't halve the bitrate
        assert size_mono == size_stereo

    def test_expected_file_size_high_quality(self) -> None:
        """Expected file size for HIGH quality 60 seconds stereo."""
        # 320 kbps * 60 seconds * 125 bytes per kilobit per second
        expected = 320 * 60 * 125
        actual = calculate_audio_file_size(60, AudioQuality.HIGH, channels=2)
        assert actual == expected

    def test_expected_file_size_mono(self) -> None:
        """Expected file size for HIGH quality 60 seconds mono."""
        # 160 kbps (half of stereo) * 60 seconds * 125 bytes
        expected = 160 * 60 * 125
        actual = calculate_audio_file_size(60, AudioQuality.HIGH, channels=1)
        assert actual == expected
