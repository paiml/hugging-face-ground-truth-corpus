"""Tests for diffusers generation functionality."""

from __future__ import annotations

import pytest

from hf_gtc.generation.diffusers import (
    GUIDANCE_SCALE_MAP,
    SUPPORTED_SIZES,
    VALID_SCHEDULER_TYPES,
    GenerationConfig,
    GenerationResult,
    SchedulerType,
    calculate_latent_size,
    create_generation_config,
    create_negative_prompt,
    estimate_vram_usage,
    get_default_negative_prompt,
    get_guidance_scale,
    get_recommended_size,
    get_recommended_steps,
    list_scheduler_types,
    validate_generation_config,
    validate_prompt,
)


class TestSchedulerType:
    """Tests for SchedulerType enum."""

    def test_euler_value(self) -> None:
        """Test EULER scheduler value."""
        assert SchedulerType.EULER.value == "euler"

    def test_ddim_value(self) -> None:
        """Test DDIM scheduler value."""
        assert SchedulerType.DDIM.value == "ddim"

    def test_dpm_solver_multistep_value(self) -> None:
        """Test DPM_SOLVER_MULTISTEP value."""
        assert SchedulerType.DPM_SOLVER_MULTISTEP.value == "dpm_solver_multistep"

    def test_all_schedulers_in_valid_set(self) -> None:
        """Test all enum values are in VALID_SCHEDULER_TYPES."""
        for scheduler in SchedulerType:
            assert scheduler.value in VALID_SCHEDULER_TYPES


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        assert config.num_inference_steps == 50
        assert config.guidance_scale == pytest.approx(7.5)
        assert config.height == 512
        assert config.width == 512
        assert config.scheduler == SchedulerType.EULER
        assert config.seed == 42
        assert config.negative_prompt is None

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(AttributeError):
            config.num_inference_steps = 100  # type: ignore[misc]


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_creation(self) -> None:
        """Test basic result creation."""
        result = GenerationResult(
            images=(b"fake_data",),
            seeds=(42,),
            nsfw_detected=(False,),
            inference_time_ms=1500.0,
        )
        assert len(result.images) == 1
        assert result.seeds == (42,)
        assert result.nsfw_detected == (False,)
        assert result.inference_time_ms == pytest.approx(1500.0)

    def test_frozen(self) -> None:
        """Test that result is immutable."""
        result = GenerationResult(
            images=(b"fake_data",),
            seeds=(42,),
            nsfw_detected=(False,),
            inference_time_ms=1500.0,
        )
        with pytest.raises(AttributeError):
            result.inference_time_ms = 2000.0  # type: ignore[misc]


class TestValidateGenerationConfig:
    """Tests for validate_generation_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        validate_generation_config(config)  # Should not raise

    def test_zero_inference_steps_raises(self) -> None:
        """Test that zero inference steps raises error."""
        config = GenerationConfig(
            num_inference_steps=0,
            guidance_scale=7.5,
            height=512,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(ValueError, match="num_inference_steps must be positive"):
            validate_generation_config(config)

    def test_negative_inference_steps_raises(self) -> None:
        """Test that negative inference steps raises error."""
        config = GenerationConfig(
            num_inference_steps=-10,
            guidance_scale=7.5,
            height=512,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(ValueError, match="num_inference_steps must be positive"):
            validate_generation_config(config)

    def test_guidance_scale_below_one_raises(self) -> None:
        """Test that guidance scale below 1 raises error."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=0.5,
            height=512,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(ValueError, match=r"guidance_scale must be >= 1\.0"):
            validate_generation_config(config)

    def test_zero_height_raises(self) -> None:
        """Test that zero height raises error."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            height=0,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(ValueError, match="height must be positive"):
            validate_generation_config(config)

    def test_zero_width_raises(self) -> None:
        """Test that zero width raises error."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=0,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(ValueError, match="width must be positive"):
            validate_generation_config(config)

    def test_height_not_divisible_by_8_raises(self) -> None:
        """Test that height not divisible by 8 raises error."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            height=513,
            width=512,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(ValueError, match="height must be divisible by 8"):
            validate_generation_config(config)

    def test_width_not_divisible_by_8_raises(self) -> None:
        """Test that width not divisible by 8 raises error."""
        config = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=510,
            scheduler=SchedulerType.EULER,
            seed=42,
            negative_prompt=None,
        )
        with pytest.raises(ValueError, match="width must be divisible by 8"):
            validate_generation_config(config)


class TestCreateGenerationConfig:
    """Tests for create_generation_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_generation_config()
        assert config.num_inference_steps == 50
        assert config.guidance_scale == pytest.approx(7.5)
        assert config.height == 512
        assert config.width == 512
        assert config.scheduler == SchedulerType.EULER
        assert config.seed is None
        assert config.negative_prompt is None

    def test_custom_inference_steps(self) -> None:
        """Test custom inference steps."""
        config = create_generation_config(num_inference_steps=20)
        assert config.num_inference_steps == 20

    def test_custom_guidance_scale(self) -> None:
        """Test custom guidance scale."""
        config = create_generation_config(guidance_scale=12.0)
        assert config.guidance_scale == pytest.approx(12.0)

    def test_custom_dimensions(self) -> None:
        """Test custom image dimensions."""
        config = create_generation_config(height=1024, width=768)
        assert config.height == 1024
        assert config.width == 768

    def test_custom_scheduler(self) -> None:
        """Test custom scheduler."""
        config = create_generation_config(scheduler="ddim")
        assert config.scheduler == SchedulerType.DDIM

    def test_with_seed(self) -> None:
        """Test configuration with seed."""
        config = create_generation_config(seed=42)
        assert config.seed == 42

    def test_with_negative_prompt(self) -> None:
        """Test configuration with negative prompt."""
        config = create_generation_config(negative_prompt="blurry, low quality")
        assert config.negative_prompt == "blurry, low quality"

    def test_invalid_scheduler_raises(self) -> None:
        """Test that invalid scheduler raises error."""
        with pytest.raises(ValueError, match="scheduler must be one of"):
            create_generation_config(scheduler="invalid_scheduler")

    def test_invalid_steps_raises(self) -> None:
        """Test that invalid steps raises error."""
        with pytest.raises(ValueError, match="num_inference_steps must be positive"):
            create_generation_config(num_inference_steps=0)


class TestGetGuidanceScale:
    """Tests for get_guidance_scale function."""

    def test_none_level(self) -> None:
        """Test 'none' guidance level."""
        assert get_guidance_scale("none") == pytest.approx(1.0)

    def test_low_level(self) -> None:
        """Test 'low' guidance level."""
        assert get_guidance_scale("low") == pytest.approx(3.5)

    def test_medium_level(self) -> None:
        """Test 'medium' guidance level."""
        assert get_guidance_scale("medium") == pytest.approx(7.5)

    def test_high_level(self) -> None:
        """Test 'high' guidance level."""
        assert get_guidance_scale("high") == pytest.approx(12.0)

    def test_very_high_level(self) -> None:
        """Test 'very_high' guidance level."""
        assert get_guidance_scale("very_high") == pytest.approx(20.0)


class TestListSchedulerTypes:
    """Tests for list_scheduler_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_scheduler_types()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_scheduler_types()
        assert result == sorted(result)

    def test_contains_euler(self) -> None:
        """Test that euler is in the list."""
        result = list_scheduler_types()
        assert "euler" in result

    def test_contains_ddim(self) -> None:
        """Test that ddim is in the list."""
        result = list_scheduler_types()
        assert "ddim" in result


class TestCalculateLatentSize:
    """Tests for calculate_latent_size function."""

    def test_standard_512(self) -> None:
        """Test calculation for 512x512."""
        assert calculate_latent_size(512, 512) == (64, 64)

    def test_standard_1024(self) -> None:
        """Test calculation for 1024x1024."""
        assert calculate_latent_size(1024, 1024) == (128, 128)

    def test_rectangular(self) -> None:
        """Test calculation for rectangular image."""
        assert calculate_latent_size(1024, 768) == (128, 96)

    def test_custom_scale_factor(self) -> None:
        """Test calculation with custom scale factor."""
        assert calculate_latent_size(512, 512, scale_factor=4) == (128, 128)

    def test_zero_height_raises(self) -> None:
        """Test that zero height raises error."""
        with pytest.raises(ValueError, match="height must be positive"):
            calculate_latent_size(0, 512)

    def test_zero_width_raises(self) -> None:
        """Test that zero width raises error."""
        with pytest.raises(ValueError, match="width must be positive"):
            calculate_latent_size(512, 0)

    def test_zero_scale_factor_raises(self) -> None:
        """Test that zero scale factor raises error."""
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            calculate_latent_size(512, 512, scale_factor=0)

    def test_height_not_divisible_raises(self) -> None:
        """Test that non-divisible height raises error."""
        with pytest.raises(ValueError, match="height must be divisible"):
            calculate_latent_size(513, 512)

    def test_width_not_divisible_raises(self) -> None:
        """Test that non-divisible width raises error."""
        with pytest.raises(ValueError, match="width must be divisible"):
            calculate_latent_size(512, 513)


class TestEstimateVramUsage:
    """Tests for estimate_vram_usage function."""

    def test_basic_512(self) -> None:
        """Test VRAM estimate for 512x512."""
        usage = estimate_vram_usage(512, 512)
        assert 2.0 < usage < 10.0  # Reasonable range

    def test_larger_image_uses_more_vram(self) -> None:
        """Test that larger images use more VRAM."""
        usage_512 = estimate_vram_usage(512, 512)
        usage_1024 = estimate_vram_usage(1024, 1024)
        assert usage_1024 > usage_512

    def test_larger_batch_uses_more_vram(self) -> None:
        """Test that larger batches use more VRAM."""
        usage_1 = estimate_vram_usage(512, 512, batch_size=1)
        usage_4 = estimate_vram_usage(512, 512, batch_size=4)
        assert usage_4 > usage_1

    def test_fp32_uses_more_than_fp16(self) -> None:
        """Test that fp32 uses more VRAM than fp16."""
        usage_fp16 = estimate_vram_usage(512, 512, precision="fp16")
        usage_fp32 = estimate_vram_usage(512, 512, precision="fp32")
        assert usage_fp32 > usage_fp16

    def test_zero_height_raises(self) -> None:
        """Test that zero height raises error."""
        with pytest.raises(ValueError, match="height must be positive"):
            estimate_vram_usage(0, 512)

    def test_zero_width_raises(self) -> None:
        """Test that zero width raises error."""
        with pytest.raises(ValueError, match="width must be positive"):
            estimate_vram_usage(512, 0)

    def test_zero_batch_size_raises(self) -> None:
        """Test that zero batch size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_vram_usage(512, 512, batch_size=0)

    def test_invalid_precision_raises(self) -> None:
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="precision must be one of"):
            estimate_vram_usage(512, 512, precision="int8")  # type: ignore[arg-type]


class TestGetRecommendedSteps:
    """Tests for get_recommended_steps function."""

    def test_sd15(self) -> None:
        """Test recommended steps for SD 1.5."""
        assert get_recommended_steps("sd15") == 50

    def test_sd21(self) -> None:
        """Test recommended steps for SD 2.1."""
        assert get_recommended_steps("sd21") == 50

    def test_sdxl(self) -> None:
        """Test recommended steps for SDXL."""
        assert get_recommended_steps("sdxl") == 30

    def test_sdxl_turbo(self) -> None:
        """Test recommended steps for SDXL Turbo."""
        assert get_recommended_steps("sdxl_turbo") == 4


class TestGetRecommendedSize:
    """Tests for get_recommended_size function."""

    def test_sd15(self) -> None:
        """Test recommended size for SD 1.5."""
        assert get_recommended_size("sd15") == (512, 512)

    def test_sd21(self) -> None:
        """Test recommended size for SD 2.1."""
        assert get_recommended_size("sd21") == (768, 768)

    def test_sdxl(self) -> None:
        """Test recommended size for SDXL."""
        assert get_recommended_size("sdxl") == (1024, 1024)

    def test_sdxl_turbo(self) -> None:
        """Test recommended size for SDXL Turbo."""
        assert get_recommended_size("sdxl_turbo") == (512, 512)


class TestValidatePrompt:
    """Tests for validate_prompt function."""

    def test_short_prompt_valid(self) -> None:
        """Test that short prompts are valid."""
        assert validate_prompt("A cat sitting on a mat") is True

    def test_empty_prompt_valid(self) -> None:
        """Test that empty prompts are valid."""
        assert validate_prompt("") is True

    def test_long_prompt_invalid(self) -> None:
        """Test that very long prompts are invalid."""
        assert validate_prompt("word " * 100) is False

    def test_custom_max_tokens(self) -> None:
        """Test validation with custom max tokens."""
        assert validate_prompt("a b c d e", max_tokens=10) is True
        assert validate_prompt("a b c d e f g h i j k l m n o", max_tokens=5) is False

    def test_none_prompt_raises(self) -> None:
        """Test that None prompt raises error."""
        with pytest.raises(ValueError, match="prompt cannot be None"):
            validate_prompt(None)  # type: ignore[arg-type]

    def test_zero_max_tokens_raises(self) -> None:
        """Test that zero max tokens raises error."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_prompt("test", max_tokens=0)


class TestCreateNegativePrompt:
    """Tests for create_negative_prompt function."""

    def test_empty_categories(self) -> None:
        """Test with empty categories."""
        assert create_negative_prompt(()) == ""

    def test_single_category(self) -> None:
        """Test with single category."""
        assert create_negative_prompt(("blurry",)) == "blurry"

    def test_multiple_categories(self) -> None:
        """Test with multiple categories."""
        result = create_negative_prompt(("blurry", "low quality"))
        assert result == "blurry, low quality"

    def test_categories_preserved(self) -> None:
        """Test that all categories are in result."""
        categories = ("nsfw", "watermark", "text")
        result = create_negative_prompt(categories)
        for cat in categories:
            assert cat in result


class TestGetDefaultNegativePrompt:
    """Tests for get_default_negative_prompt function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = get_default_negative_prompt()
        assert isinstance(result, str)

    def test_contains_blurry(self) -> None:
        """Test that 'blurry' is in default prompt."""
        result = get_default_negative_prompt()
        assert "blurry" in result

    def test_contains_low_quality(self) -> None:
        """Test that 'low quality' is in default prompt."""
        result = get_default_negative_prompt()
        assert "low quality" in result

    def test_contains_watermark(self) -> None:
        """Test that 'watermark' is in default prompt."""
        result = get_default_negative_prompt()
        assert "watermark" in result

    def test_not_empty(self) -> None:
        """Test that default prompt is not empty."""
        result = get_default_negative_prompt()
        assert len(result) > 0


class TestConstants:
    """Tests for module constants."""

    def test_supported_sizes(self) -> None:
        """Test SUPPORTED_SIZES contains expected values."""
        assert 512 in SUPPORTED_SIZES
        assert 1024 in SUPPORTED_SIZES

    def test_guidance_scale_map_keys(self) -> None:
        """Test GUIDANCE_SCALE_MAP has expected keys."""
        assert "none" in GUIDANCE_SCALE_MAP
        assert "low" in GUIDANCE_SCALE_MAP
        assert "medium" in GUIDANCE_SCALE_MAP
        assert "high" in GUIDANCE_SCALE_MAP
        assert "very_high" in GUIDANCE_SCALE_MAP

    def test_valid_scheduler_types_not_empty(self) -> None:
        """Test VALID_SCHEDULER_TYPES is not empty."""
        assert len(VALID_SCHEDULER_TYPES) > 0
