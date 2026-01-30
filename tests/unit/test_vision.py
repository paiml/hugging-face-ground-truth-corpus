"""Tests for vision and multimodal utilities."""

from __future__ import annotations

import pytest

from hf_gtc.multimodal.vision import (
    IMAGE_SIZE_PRESETS,
    VALID_MODALITIES,
    VALID_VISION_MODELS,
    ImageConfig,
    ImageInput,
    ImageProcessor,
    ModalityType,
    MultimodalConfig,
    MultimodalInput,
    VisionEncoderConfig,
    VisionModelType,
    calculate_image_tokens,
    calculate_patch_count,
    create_image_config,
    create_image_input,
    create_multimodal_config,
    create_vision_encoder_config,
    estimate_multimodal_memory,
    format_image_size,
    get_default_image_size,
    get_modality_type,
    get_recommended_processor,
    get_vision_model_type,
    list_modality_types,
    list_vision_model_types,
    validate_image_config,
    validate_image_dimensions,
    validate_multimodal_config,
)


class TestVisionModelType:
    """Tests for VisionModelType enum."""

    def test_clip_value(self) -> None:
        """Test CLIP value."""
        assert VisionModelType.CLIP.value == "clip"

    def test_siglip_value(self) -> None:
        """Test SigLIP value."""
        assert VisionModelType.SIGLIP.value == "siglip"

    def test_vit_value(self) -> None:
        """Test ViT value."""
        assert VisionModelType.VIT.value == "vit"

    def test_dino_value(self) -> None:
        """Test DINO value."""
        assert VisionModelType.DINO.value == "dino"

    def test_eva_value(self) -> None:
        """Test EVA value."""
        assert VisionModelType.EVA.value == "eva"

    def test_convnext_value(self) -> None:
        """Test ConvNeXt value."""
        assert VisionModelType.CONVNEXT.value == "convnext"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_VISION_MODELS."""
        for model in VisionModelType:
            assert model.value in VALID_VISION_MODELS


class TestModalityType:
    """Tests for ModalityType enum."""

    def test_text_value(self) -> None:
        """Test TEXT value."""
        assert ModalityType.TEXT.value == "text"

    def test_image_value(self) -> None:
        """Test IMAGE value."""
        assert ModalityType.IMAGE.value == "image"

    def test_audio_value(self) -> None:
        """Test AUDIO value."""
        assert ModalityType.AUDIO.value == "audio"

    def test_video_value(self) -> None:
        """Test VIDEO value."""
        assert ModalityType.VIDEO.value == "video"

    def test_text_image_value(self) -> None:
        """Test TEXT_IMAGE value."""
        assert ModalityType.TEXT_IMAGE.value == "text_image"

    def test_text_audio_value(self) -> None:
        """Test TEXT_AUDIO value."""
        assert ModalityType.TEXT_AUDIO.value == "text_audio"

    def test_text_video_value(self) -> None:
        """Test TEXT_VIDEO value."""
        assert ModalityType.TEXT_VIDEO.value == "text_video"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_MODALITIES."""
        for modality in ModalityType:
            assert modality.value in VALID_MODALITIES


class TestImageConfig:
    """Tests for ImageConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating image config."""
        config = ImageConfig(
            width=224,
            height=224,
            channels=3,
            normalize=True,
            resize_mode="crop",
        )
        assert config.width == 224
        assert config.height == 224
        assert config.channels == 3
        assert config.normalize is True
        assert config.resize_mode == "crop"

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = ImageConfig(224, 224, 3, True, "crop")
        with pytest.raises(AttributeError):
            config.width = 336  # type: ignore[misc]


class TestVisionEncoderConfig:
    """Tests for VisionEncoderConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating vision encoder config."""
        config = VisionEncoderConfig(
            model_type=VisionModelType.CLIP,
            patch_size=14,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
        )
        assert config.model_type == VisionModelType.CLIP
        assert config.patch_size == 14
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = VisionEncoderConfig(VisionModelType.VIT, 16, 768, 12, 12)
        with pytest.raises(AttributeError):
            config.patch_size = 14  # type: ignore[misc]


class TestImageInput:
    """Tests for ImageInput dataclass."""

    def test_create_input(self) -> None:
        """Test creating image input."""
        img = ImageInput(
            path="image.png",
            width=1024,
            height=768,
            format="png",
        )
        assert img.path == "image.png"
        assert img.width == 1024
        assert img.height == 768
        assert img.format == "png"

    def test_frozen(self) -> None:
        """Test input is immutable."""
        img = ImageInput("test.jpg", 640, 480, "jpg")
        with pytest.raises(AttributeError):
            img.path = "new.jpg"  # type: ignore[misc]


class TestMultimodalInput:
    """Tests for MultimodalInput dataclass."""

    def test_create_input(self) -> None:
        """Test creating multimodal input."""
        inp = MultimodalInput(
            text="Describe this image",
            images=(),
            modality=ModalityType.TEXT_IMAGE,
        )
        assert inp.text == "Describe this image"
        assert inp.images == ()
        assert inp.modality == ModalityType.TEXT_IMAGE

    def test_with_images(self) -> None:
        """Test input with images."""
        img = ImageInput("test.png", 224, 224, "png")
        inp = MultimodalInput(
            text="What do you see?",
            images=(img,),
            modality=ModalityType.TEXT_IMAGE,
        )
        assert len(inp.images) == 1

    def test_frozen(self) -> None:
        """Test input is immutable."""
        inp = MultimodalInput("text", (), ModalityType.TEXT)
        with pytest.raises(AttributeError):
            inp.text = "new text"  # type: ignore[misc]


class TestMultimodalConfig:
    """Tests for MultimodalConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating multimodal config."""
        img_cfg = ImageConfig(224, 224, 3, True, "crop")
        vis_cfg = VisionEncoderConfig(VisionModelType.CLIP, 14, 768, 12, 12)
        config = MultimodalConfig(
            image_config=img_cfg,
            vision_encoder=vis_cfg,
            max_images=5,
            modality=ModalityType.TEXT_IMAGE,
        )
        assert config.max_images == 5
        assert config.modality == ModalityType.TEXT_IMAGE

    def test_frozen(self) -> None:
        """Test config is immutable."""
        img_cfg = ImageConfig(224, 224, 3, True, "crop")
        vis_cfg = VisionEncoderConfig(VisionModelType.CLIP, 14, 768, 12, 12)
        config = MultimodalConfig(img_cfg, vis_cfg, 5, ModalityType.TEXT_IMAGE)
        with pytest.raises(AttributeError):
            config.max_images = 10  # type: ignore[misc]


class TestImageProcessor:
    """Tests for ImageProcessor dataclass."""

    def test_create_processor(self) -> None:
        """Test creating image processor."""
        proc = ImageProcessor(
            name="clip",
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            do_resize=True,
            do_normalize=True,
        )
        assert proc.name == "clip"
        assert len(proc.image_mean) == 3
        assert proc.do_resize is True

    def test_frozen(self) -> None:
        """Test processor is immutable."""
        proc = ImageProcessor("vit", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True, True)
        with pytest.raises(AttributeError):
            proc.name = "new"  # type: ignore[misc]


class TestValidateImageDimensions:
    """Tests for validate_image_dimensions function."""

    def test_valid_dimensions(self) -> None:
        """Test valid dimensions."""
        validate_image_dimensions(224, 224)  # Should not raise

    def test_zero_width(self) -> None:
        """Test zero width."""
        with pytest.raises(ValueError, match="width must be positive"):
            validate_image_dimensions(0, 224)

    def test_negative_width(self) -> None:
        """Test negative width."""
        with pytest.raises(ValueError, match="width must be positive"):
            validate_image_dimensions(-1, 224)

    def test_zero_height(self) -> None:
        """Test zero height."""
        with pytest.raises(ValueError, match="height must be positive"):
            validate_image_dimensions(224, 0)

    def test_negative_height(self) -> None:
        """Test negative height."""
        with pytest.raises(ValueError, match="height must be positive"):
            validate_image_dimensions(224, -1)


class TestValidateImageConfig:
    """Tests for validate_image_config function."""

    def test_valid_config(self) -> None:
        """Test valid config."""
        config = ImageConfig(224, 224, 3, True, "crop")
        validate_image_config(config)  # Should not raise

    def test_invalid_width(self) -> None:
        """Test invalid width."""
        config = ImageConfig(-1, 224, 3, True, "crop")
        with pytest.raises(ValueError, match="width must be positive"):
            validate_image_config(config)

    def test_invalid_channels(self) -> None:
        """Test invalid channels."""
        config = ImageConfig(224, 224, 0, True, "crop")
        with pytest.raises(ValueError, match="channels must be positive"):
            validate_image_config(config)

    def test_invalid_resize_mode(self) -> None:
        """Test invalid resize mode."""
        config = ImageConfig(224, 224, 3, True, "invalid")
        with pytest.raises(ValueError, match="resize_mode must be one of"):
            validate_image_config(config)

    def test_valid_resize_modes(self) -> None:
        """Test all valid resize modes."""
        for mode in ("crop", "pad", "stretch"):
            config = ImageConfig(224, 224, 3, True, mode)
            validate_image_config(config)  # Should not raise


class TestValidateMultimodalConfig:
    """Tests for validate_multimodal_config function."""

    def test_valid_config(self) -> None:
        """Test valid config."""
        img_cfg = ImageConfig(224, 224, 3, True, "crop")
        vis_cfg = VisionEncoderConfig(VisionModelType.CLIP, 14, 768, 12, 12)
        config = MultimodalConfig(img_cfg, vis_cfg, 5, ModalityType.TEXT_IMAGE)
        validate_multimodal_config(config)  # Should not raise

    def test_invalid_image_config(self) -> None:
        """Test invalid image config."""
        img_cfg = ImageConfig(-1, 224, 3, True, "crop")
        vis_cfg = VisionEncoderConfig(VisionModelType.CLIP, 14, 768, 12, 12)
        config = MultimodalConfig(img_cfg, vis_cfg, 5, ModalityType.TEXT_IMAGE)
        with pytest.raises(ValueError, match="width must be positive"):
            validate_multimodal_config(config)

    def test_invalid_max_images(self) -> None:
        """Test invalid max_images."""
        img_cfg = ImageConfig(224, 224, 3, True, "crop")
        vis_cfg = VisionEncoderConfig(VisionModelType.CLIP, 14, 768, 12, 12)
        config = MultimodalConfig(img_cfg, vis_cfg, 0, ModalityType.TEXT_IMAGE)
        with pytest.raises(ValueError, match="max_images must be positive"):
            validate_multimodal_config(config)

    def test_invalid_patch_size(self) -> None:
        """Test invalid patch_size."""
        img_cfg = ImageConfig(224, 224, 3, True, "crop")
        vis_cfg = VisionEncoderConfig(VisionModelType.CLIP, 0, 768, 12, 12)
        config = MultimodalConfig(img_cfg, vis_cfg, 5, ModalityType.TEXT_IMAGE)
        with pytest.raises(ValueError, match="patch_size must be positive"):
            validate_multimodal_config(config)


class TestCreateImageConfig:
    """Tests for create_image_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_image_config()
        assert config.width == 224
        assert config.height == 224
        assert config.channels == 3
        assert config.normalize is True
        assert config.resize_mode == "crop"

    def test_custom_dimensions(self) -> None:
        """Test custom dimensions."""
        config = create_image_config(width=336, height=336)
        assert config.width == 336
        assert config.height == 336

    def test_custom_channels(self) -> None:
        """Test custom channels."""
        config = create_image_config(channels=1)
        assert config.channels == 1

    def test_no_normalize(self) -> None:
        """Test normalize=False."""
        config = create_image_config(normalize=False)
        assert config.normalize is False

    def test_pad_resize_mode(self) -> None:
        """Test pad resize mode."""
        config = create_image_config(resize_mode="pad")
        assert config.resize_mode == "pad"

    def test_stretch_resize_mode(self) -> None:
        """Test stretch resize mode."""
        config = create_image_config(resize_mode="stretch")
        assert config.resize_mode == "stretch"

    def test_invalid_width(self) -> None:
        """Test invalid width."""
        with pytest.raises(ValueError, match="width must be positive"):
            create_image_config(width=-1)

    def test_invalid_resize_mode(self) -> None:
        """Test invalid resize mode."""
        with pytest.raises(ValueError, match="resize_mode must be one of"):
            create_image_config(resize_mode="invalid")


class TestCreateVisionEncoderConfig:
    """Tests for create_vision_encoder_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_vision_encoder_config()
        assert config.model_type == VisionModelType.CLIP
        assert config.patch_size == 14
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12

    def test_vit_model(self) -> None:
        """Test ViT model."""
        config = create_vision_encoder_config(model_type="vit")
        assert config.model_type == VisionModelType.VIT

    def test_siglip_model(self) -> None:
        """Test SigLIP model."""
        config = create_vision_encoder_config(model_type="siglip")
        assert config.model_type == VisionModelType.SIGLIP

    def test_dino_model(self) -> None:
        """Test DINO model."""
        config = create_vision_encoder_config(model_type="dino")
        assert config.model_type == VisionModelType.DINO

    def test_eva_model(self) -> None:
        """Test EVA model."""
        config = create_vision_encoder_config(model_type="eva")
        assert config.model_type == VisionModelType.EVA

    def test_convnext_model(self) -> None:
        """Test ConvNeXt model."""
        config = create_vision_encoder_config(model_type="convnext")
        assert config.model_type == VisionModelType.CONVNEXT

    def test_custom_patch_size(self) -> None:
        """Test custom patch size."""
        config = create_vision_encoder_config(patch_size=16)
        assert config.patch_size == 16

    def test_custom_hidden_size(self) -> None:
        """Test custom hidden size."""
        config = create_vision_encoder_config(hidden_size=1024)
        assert config.hidden_size == 1024

    def test_custom_num_layers(self) -> None:
        """Test custom num_layers."""
        config = create_vision_encoder_config(num_layers=24)
        assert config.num_layers == 24

    def test_custom_num_heads(self) -> None:
        """Test custom num_heads."""
        config = create_vision_encoder_config(num_heads=16)
        assert config.num_heads == 16

    def test_invalid_model_type(self) -> None:
        """Test invalid model type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_vision_encoder_config(model_type="invalid")

    def test_invalid_patch_size(self) -> None:
        """Test invalid patch size."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            create_vision_encoder_config(patch_size=0)

    def test_invalid_hidden_size(self) -> None:
        """Test invalid hidden size."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            create_vision_encoder_config(hidden_size=0)

    def test_invalid_num_layers(self) -> None:
        """Test invalid num_layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            create_vision_encoder_config(num_layers=0)

    def test_invalid_num_heads(self) -> None:
        """Test invalid num_heads."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            create_vision_encoder_config(num_heads=0)


class TestCreateImageInput:
    """Tests for create_image_input function."""

    def test_basic_input(self) -> None:
        """Test basic image input."""
        img = create_image_input("test.png", 1024, 768)
        assert img.path == "test.png"
        assert img.width == 1024
        assert img.height == 768
        assert img.format == "png"

    def test_jpg_format(self) -> None:
        """Test JPG format."""
        img = create_image_input("test.jpg", 640, 480, format="jpg")
        assert img.format == "jpg"

    def test_jpeg_format(self) -> None:
        """Test JPEG format."""
        img = create_image_input("test.jpeg", 640, 480, format="jpeg")
        assert img.format == "jpeg"

    def test_gif_format(self) -> None:
        """Test GIF format."""
        img = create_image_input("test.gif", 100, 100, format="gif")
        assert img.format == "gif"

    def test_bmp_format(self) -> None:
        """Test BMP format."""
        img = create_image_input("test.bmp", 100, 100, format="bmp")
        assert img.format == "bmp"

    def test_webp_format(self) -> None:
        """Test WebP format."""
        img = create_image_input("test.webp", 100, 100, format="webp")
        assert img.format == "webp"

    def test_format_lowercase(self) -> None:
        """Test format is lowercased."""
        img = create_image_input("test.PNG", 100, 100, format="PNG")
        assert img.format == "png"

    def test_empty_path(self) -> None:
        """Test empty path."""
        with pytest.raises(ValueError, match="path cannot be empty"):
            create_image_input("", 1024, 768)

    def test_invalid_width(self) -> None:
        """Test invalid width."""
        with pytest.raises(ValueError, match="width must be positive"):
            create_image_input("test.png", 0, 768)

    def test_invalid_height(self) -> None:
        """Test invalid height."""
        with pytest.raises(ValueError, match="height must be positive"):
            create_image_input("test.png", 1024, 0)

    def test_invalid_format(self) -> None:
        """Test invalid format."""
        with pytest.raises(ValueError, match="format must be one of"):
            create_image_input("test.xyz", 100, 100, format="xyz")


class TestCreateMultimodalConfig:
    """Tests for create_multimodal_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_multimodal_config()
        assert config.image_config.width == 224
        assert config.image_config.height == 224
        assert config.vision_encoder.model_type == VisionModelType.CLIP
        assert config.vision_encoder.patch_size == 14
        assert config.max_images == 5
        assert config.modality == ModalityType.TEXT_IMAGE

    def test_custom_image_size(self) -> None:
        """Test custom image size."""
        config = create_multimodal_config(image_width=336, image_height=336)
        assert config.image_config.width == 336
        assert config.image_config.height == 336

    def test_custom_vision_model(self) -> None:
        """Test custom vision model."""
        config = create_multimodal_config(vision_model="vit")
        assert config.vision_encoder.model_type == VisionModelType.VIT

    def test_custom_patch_size(self) -> None:
        """Test custom patch size."""
        config = create_multimodal_config(patch_size=16)
        assert config.vision_encoder.patch_size == 16

    def test_custom_max_images(self) -> None:
        """Test custom max_images."""
        config = create_multimodal_config(max_images=10)
        assert config.max_images == 10

    def test_invalid_max_images(self) -> None:
        """Test invalid max_images."""
        with pytest.raises(ValueError, match="max_images must be positive"):
            create_multimodal_config(max_images=0)

    def test_invalid_vision_model(self) -> None:
        """Test invalid vision model."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_multimodal_config(vision_model="invalid")


class TestListVisionModelTypes:
    """Tests for list_vision_model_types function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        types = list_vision_model_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_vision_model_types()
        assert "clip" in types
        assert "vit" in types
        assert "dino" in types

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        types = list_vision_model_types()
        assert types == sorted(types)


class TestListModalityTypes:
    """Tests for list_modality_types function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        types = list_modality_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_modality_types()
        assert "text_image" in types
        assert "text" in types
        assert "image" in types

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        types = list_modality_types()
        assert types == sorted(types)


class TestGetVisionModelType:
    """Tests for get_vision_model_type function."""

    def test_clip(self) -> None:
        """Test getting CLIP type."""
        assert get_vision_model_type("clip") == VisionModelType.CLIP

    def test_vit(self) -> None:
        """Test getting ViT type."""
        assert get_vision_model_type("vit") == VisionModelType.VIT

    def test_dino(self) -> None:
        """Test getting DINO type."""
        assert get_vision_model_type("dino") == VisionModelType.DINO

    def test_invalid(self) -> None:
        """Test invalid model type."""
        with pytest.raises(ValueError, match="model type must be one of"):
            get_vision_model_type("invalid")


class TestGetModalityType:
    """Tests for get_modality_type function."""

    def test_text_image(self) -> None:
        """Test getting TEXT_IMAGE type."""
        assert get_modality_type("text_image") == ModalityType.TEXT_IMAGE

    def test_text(self) -> None:
        """Test getting TEXT type."""
        assert get_modality_type("text") == ModalityType.TEXT

    def test_image(self) -> None:
        """Test getting IMAGE type."""
        assert get_modality_type("image") == ModalityType.IMAGE

    def test_invalid(self) -> None:
        """Test invalid modality type."""
        with pytest.raises(ValueError, match="modality must be one of"):
            get_modality_type("invalid")


class TestGetDefaultImageSize:
    """Tests for get_default_image_size function."""

    def test_small(self) -> None:
        """Test small preset."""
        assert get_default_image_size("small") == (224, 224)

    def test_medium(self) -> None:
        """Test medium preset."""
        assert get_default_image_size("medium") == (336, 336)

    def test_large(self) -> None:
        """Test large preset."""
        assert get_default_image_size("large") == (448, 448)

    def test_xlarge(self) -> None:
        """Test xlarge preset."""
        assert get_default_image_size("xlarge") == (672, 672)

    def test_all_presets(self) -> None:
        """Test all presets are in IMAGE_SIZE_PRESETS."""
        for preset in ("small", "medium", "large", "xlarge"):
            size = get_default_image_size(preset)  # type: ignore[arg-type]
            assert size == IMAGE_SIZE_PRESETS[preset]  # type: ignore[index]


class TestCalculatePatchCount:
    """Tests for calculate_patch_count function."""

    def test_224_patch14(self) -> None:
        """Test 224x224 with patch 14."""
        count = calculate_patch_count(224, 224, 14)
        assert count == 256  # 16 * 16

    def test_336_patch14(self) -> None:
        """Test 336x336 with patch 14."""
        count = calculate_patch_count(336, 336, 14)
        assert count == 576  # 24 * 24

    def test_224_patch16(self) -> None:
        """Test 224x224 with patch 16."""
        count = calculate_patch_count(224, 224, 16)
        assert count == 196  # 14 * 14

    def test_rectangular_image(self) -> None:
        """Test rectangular image."""
        count = calculate_patch_count(224, 112, 14)
        assert count == 128  # 16 * 8

    def test_invalid_width(self) -> None:
        """Test invalid width."""
        with pytest.raises(ValueError, match="width must be positive"):
            calculate_patch_count(0, 224, 14)

    def test_invalid_patch_size(self) -> None:
        """Test invalid patch size."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            calculate_patch_count(224, 224, 0)


class TestCalculateImageTokens:
    """Tests for calculate_image_tokens function."""

    def test_with_cls(self) -> None:
        """Test with CLS token."""
        tokens = calculate_image_tokens(224, 224, 14, include_cls=True)
        assert tokens == 257  # 256 + 1

    def test_without_cls(self) -> None:
        """Test without CLS token."""
        tokens = calculate_image_tokens(224, 224, 14, include_cls=False)
        assert tokens == 256

    def test_larger_image(self) -> None:
        """Test larger image."""
        tokens = calculate_image_tokens(336, 336, 14, include_cls=True)
        assert tokens == 577  # 576 + 1


class TestEstimateMultimodalMemory:
    """Tests for estimate_multimodal_memory function."""

    def test_single_image(self) -> None:
        """Test single image memory."""
        mem = estimate_multimodal_memory(1, 224, 224)
        assert mem > 0
        # 257 tokens * 768 hidden * 2 bytes = 394752
        assert mem == 257 * 768 * 2

    def test_multiple_images(self) -> None:
        """Test multiple images memory."""
        mem_1 = estimate_multimodal_memory(1, 224, 224)
        mem_2 = estimate_multimodal_memory(2, 224, 224)
        assert mem_2 == mem_1 * 2

    def test_larger_image(self) -> None:
        """Test larger image uses more memory."""
        mem_224 = estimate_multimodal_memory(1, 224, 224)
        mem_336 = estimate_multimodal_memory(1, 336, 336)
        assert mem_336 > mem_224

    def test_custom_hidden_size(self) -> None:
        """Test custom hidden size."""
        mem_768 = estimate_multimodal_memory(1, 224, 224, hidden_size=768)
        mem_1024 = estimate_multimodal_memory(1, 224, 224, hidden_size=1024)
        assert mem_1024 > mem_768

    def test_custom_dtype_bytes(self) -> None:
        """Test custom dtype_bytes."""
        mem_fp16 = estimate_multimodal_memory(1, 224, 224, dtype_bytes=2)
        mem_fp32 = estimate_multimodal_memory(1, 224, 224, dtype_bytes=4)
        assert mem_fp32 == mem_fp16 * 2

    def test_invalid_num_images(self) -> None:
        """Test invalid num_images."""
        with pytest.raises(ValueError, match="num_images must be positive"):
            estimate_multimodal_memory(0, 224, 224)


class TestFormatImageSize:
    """Tests for format_image_size function."""

    def test_square_image(self) -> None:
        """Test square image."""
        assert format_image_size(224, 224) == "224x224"

    def test_rectangular_image(self) -> None:
        """Test rectangular image."""
        assert format_image_size(1920, 1080) == "1920x1080"

    def test_small_image(self) -> None:
        """Test small image."""
        assert format_image_size(64, 64) == "64x64"


class TestGetRecommendedProcessor:
    """Tests for get_recommended_processor function."""

    def test_clip_processor(self) -> None:
        """Test CLIP processor."""
        proc = get_recommended_processor(VisionModelType.CLIP)
        assert proc.name == "clip"
        assert proc.do_resize is True
        assert proc.do_normalize is True

    def test_siglip_processor(self) -> None:
        """Test SigLIP processor."""
        proc = get_recommended_processor(VisionModelType.SIGLIP)
        assert proc.name == "siglip"

    def test_vit_processor(self) -> None:
        """Test ViT processor."""
        proc = get_recommended_processor(VisionModelType.VIT)
        assert proc.name == "vit"

    def test_dino_processor(self) -> None:
        """Test DINO processor."""
        proc = get_recommended_processor(VisionModelType.DINO)
        assert proc.name == "dino"

    def test_eva_processor(self) -> None:
        """Test EVA processor."""
        proc = get_recommended_processor(VisionModelType.EVA)
        assert proc.name == "eva"

    def test_convnext_processor(self) -> None:
        """Test ConvNeXt processor."""
        proc = get_recommended_processor(VisionModelType.CONVNEXT)
        assert proc.name == "convnext"

    def test_all_processors_have_mean_std(self) -> None:
        """Test all processors have image_mean and image_std."""
        for model_type in VisionModelType:
            proc = get_recommended_processor(model_type)
            assert len(proc.image_mean) == 3
            assert len(proc.image_std) == 3
