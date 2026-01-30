"""Tests for HuggingFace Spaces API functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hf_gtc.hub.spaces import (
    SpaceHardware,
    SpaceInfo,
    SpaceRuntime,
    SpaceSDK,
    SpaceStage,
    _convert_space_info,
    get_space_info,
    get_space_runtime,
    iter_spaces,
    list_hardware_tiers,
    list_sdks,
    search_spaces,
    validate_hardware,
    validate_sdk,
)


class TestSpaceSDK:
    """Tests for SpaceSDK enum."""

    def test_gradio_value(self) -> None:
        """Test GRADIO SDK value."""
        assert SpaceSDK.GRADIO.value == "gradio"

    def test_streamlit_value(self) -> None:
        """Test STREAMLIT SDK value."""
        assert SpaceSDK.STREAMLIT.value == "streamlit"

    def test_docker_value(self) -> None:
        """Test DOCKER SDK value."""
        assert SpaceSDK.DOCKER.value == "docker"

    def test_static_value(self) -> None:
        """Test STATIC SDK value."""
        assert SpaceSDK.STATIC.value == "static"


class TestSpaceStage:
    """Tests for SpaceStage enum."""

    def test_running_value(self) -> None:
        """Test RUNNING stage value."""
        assert SpaceStage.RUNNING.value == "RUNNING"

    def test_paused_value(self) -> None:
        """Test PAUSED stage value."""
        assert SpaceStage.PAUSED.value == "PAUSED"

    def test_stopped_value(self) -> None:
        """Test STOPPED stage value."""
        assert SpaceStage.STOPPED.value == "STOPPED"

    def test_building_value(self) -> None:
        """Test BUILDING stage value."""
        assert SpaceStage.BUILDING.value == "BUILDING"

    def test_no_app_file_value(self) -> None:
        """Test NO_APP_FILE stage value."""
        assert SpaceStage.NO_APP_FILE.value == "NO_APP_FILE"

    def test_config_error_value(self) -> None:
        """Test CONFIG_ERROR stage value."""
        assert SpaceStage.CONFIG_ERROR.value == "CONFIG_ERROR"


class TestSpaceHardware:
    """Tests for SpaceHardware enum."""

    def test_cpu_basic_value(self) -> None:
        """Test CPU_BASIC hardware value."""
        assert SpaceHardware.CPU_BASIC.value == "cpu-basic"

    def test_cpu_upgrade_value(self) -> None:
        """Test CPU_UPGRADE hardware value."""
        assert SpaceHardware.CPU_UPGRADE.value == "cpu-upgrade"

    def test_t4_small_value(self) -> None:
        """Test T4_SMALL hardware value."""
        assert SpaceHardware.T4_SMALL.value == "t4-small"

    def test_t4_medium_value(self) -> None:
        """Test T4_MEDIUM hardware value."""
        assert SpaceHardware.T4_MEDIUM.value == "t4-medium"

    def test_a10g_small_value(self) -> None:
        """Test A10G_SMALL hardware value."""
        assert SpaceHardware.A10G_SMALL.value == "a10g-small"

    def test_a10g_large_value(self) -> None:
        """Test A10G_LARGE hardware value."""
        assert SpaceHardware.A10G_LARGE.value == "a10g-large"

    def test_a100_large_value(self) -> None:
        """Test A100_LARGE hardware value."""
        assert SpaceHardware.A100_LARGE.value == "a100-large"


class TestSpaceInfo:
    """Tests for SpaceInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating SpaceInfo instance."""
        info = SpaceInfo(
            space_id="test/space",
            author="test",
            title="Test Space",
            sdk="gradio",
            likes=100,
            emoji="🚀",
            hardware="cpu-basic",
        )
        assert info.space_id == "test/space"
        assert info.author == "test"
        assert info.title == "Test Space"
        assert info.sdk == "gradio"
        assert info.likes == 100
        assert info.emoji == "🚀"
        assert info.hardware == "cpu-basic"

    def test_frozen(self) -> None:
        """Test that SpaceInfo is immutable."""
        info = SpaceInfo(
            space_id="test/space",
            author="test",
            title="Test",
            sdk="gradio",
            likes=0,
            emoji=None,
            hardware=None,
        )
        with pytest.raises(AttributeError):
            info.space_id = "new/space"  # type: ignore[misc]

    def test_none_values(self) -> None:
        """Test SpaceInfo with None values."""
        info = SpaceInfo(
            space_id="test/space",
            author="test",
            title=None,
            sdk=None,
            likes=0,
            emoji=None,
            hardware=None,
        )
        assert info.title is None
        assert info.sdk is None
        assert info.emoji is None
        assert info.hardware is None


class TestSpaceRuntime:
    """Tests for SpaceRuntime dataclass."""

    def test_creation(self) -> None:
        """Test creating SpaceRuntime instance."""
        runtime = SpaceRuntime(
            stage="RUNNING",
            hardware="cpu-basic",
            sdk_version="3.50.0",
        )
        assert runtime.stage == "RUNNING"
        assert runtime.hardware == "cpu-basic"
        assert runtime.sdk_version == "3.50.0"

    def test_frozen(self) -> None:
        """Test that SpaceRuntime is immutable."""
        runtime = SpaceRuntime(
            stage="RUNNING",
            hardware="cpu-basic",
            sdk_version="3.50.0",
        )
        with pytest.raises(AttributeError):
            runtime.stage = "PAUSED"  # type: ignore[misc]

    def test_none_values(self) -> None:
        """Test SpaceRuntime with None values."""
        runtime = SpaceRuntime(
            stage=None,
            hardware=None,
            sdk_version=None,
        )
        assert runtime.stage is None
        assert runtime.hardware is None
        assert runtime.sdk_version is None


class TestConvertSpaceInfo:
    """Tests for _convert_space_info function."""

    def test_basic_conversion(self) -> None:
        """Test basic Space info conversion."""
        mock = MagicMock()
        mock.id = "test/space"
        mock.author = "test"
        mock.title = "Test Space"
        mock.sdk = "gradio"
        mock.likes = 50
        mock.emoji = "🤗"
        mock.hardware = "cpu-basic"

        result = _convert_space_info(mock)

        assert result.space_id == "test/space"
        assert result.author == "test"
        assert result.title == "Test Space"
        assert result.sdk == "gradio"
        assert result.likes == 50
        assert result.emoji == "🤗"
        assert result.hardware == "cpu-basic"

    def test_none_likes(self) -> None:
        """Test conversion with None likes."""
        mock = MagicMock()
        mock.id = "test/space"
        mock.author = "test"
        mock.sdk = "gradio"
        mock.likes = None

        result = _convert_space_info(mock)

        assert result.likes == 0

    def test_missing_attributes(self) -> None:
        """Test conversion with missing optional attributes."""
        mock = MagicMock(spec=["id", "author", "sdk", "likes"])
        mock.id = "test/space"
        mock.author = "test"
        mock.sdk = "gradio"
        mock.likes = 10

        result = _convert_space_info(mock)

        assert result.space_id == "test/space"
        assert result.title is None
        assert result.emoji is None
        assert result.hardware is None


class TestSearchSpaces:
    """Tests for search_spaces function."""

    def test_limit_zero_raises_error(self) -> None:
        """Test that zero limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be between 1 and 100"):
            search_spaces(limit=0)

    def test_limit_over_100_raises_error(self) -> None:
        """Test that limit over 100 raises ValueError."""
        with pytest.raises(ValueError, match="limit must be between 1 and 100"):
            search_spaces(limit=101)

    def test_invalid_sort_raises_error(self) -> None:
        """Test that invalid sort raises ValueError."""
        with pytest.raises(ValueError, match="sort must be one of"):
            search_spaces(sort="invalid")

    def test_invalid_sdk_raises_error(self) -> None:
        """Test that invalid SDK raises ValueError."""
        with pytest.raises(ValueError, match="sdk must be one of"):
            search_spaces(sdk="invalid")

    def test_valid_sorts(self) -> None:
        """Test all valid sort options are accepted."""
        for sort in ["likes", "created", "lastModified"]:
            # Should not raise - mock the API call
            with patch("huggingface_hub.HfApi") as mock_api:
                mock_api.return_value.list_spaces.return_value = []
                search_spaces(sort=sort)

    def test_valid_sdks(self) -> None:
        """Test all valid SDK options are accepted."""
        for sdk in ["gradio", "streamlit", "docker", "static"]:
            with patch("huggingface_hub.HfApi") as mock_api:
                mock_api.return_value.list_spaces.return_value = []
                search_spaces(sdk=sdk)

    @patch("huggingface_hub.HfApi")
    def test_returns_space_info_list(self, mock_api_class: MagicMock) -> None:
        """Test that search_spaces returns list of SpaceInfo."""
        mock_space = MagicMock()
        mock_space.id = "test/space"
        mock_space.author = "test"
        mock_space.sdk = "gradio"
        mock_space.likes = 100
        mock_space.title = "Test"
        mock_space.emoji = "🚀"
        mock_space.hardware = "cpu-basic"

        mock_api_class.return_value.list_spaces.return_value = [mock_space]

        results = search_spaces(query="test")

        assert len(results) == 1
        assert isinstance(results[0], SpaceInfo)
        assert results[0].space_id == "test/space"

    @patch("huggingface_hub.HfApi")
    def test_passes_parameters_to_api(self, mock_api_class: MagicMock) -> None:
        """Test that parameters are passed correctly to HfApi."""
        mock_api_class.return_value.list_spaces.return_value = []

        search_spaces(
            query="test",
            sdk="gradio",
            author="testuser",
            limit=20,
            sort="likes",
        )

        mock_api_class.return_value.list_spaces.assert_called_once_with(
            search="test",
            author="testuser",
            limit=20,
            sort="likes",
            direction=-1,
            filter="gradio",
        )


class TestIterSpaces:
    """Tests for iter_spaces function."""

    def test_returns_iterator(self) -> None:
        """Test that iter_spaces returns an iterator."""
        with patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value.list_spaces.return_value = iter([])
            it = iter_spaces()
            assert hasattr(it, "__iter__")
            assert hasattr(it, "__next__")

    def test_invalid_sdk_raises_error(self) -> None:
        """Test that invalid SDK raises ValueError."""
        with pytest.raises(ValueError, match="sdk must be one of"):
            # Need to consume the iterator to trigger validation
            list(iter_spaces(sdk="invalid"))

    @patch("huggingface_hub.HfApi")
    def test_yields_space_info(self, mock_api_class: MagicMock) -> None:
        """Test that iter_spaces yields SpaceInfo objects."""
        mock_space = MagicMock()
        mock_space.id = "test/space"
        mock_space.author = "test"
        mock_space.sdk = "gradio"
        mock_space.likes = 50
        mock_space.title = "Test"
        mock_space.emoji = "🎉"
        mock_space.hardware = "cpu-basic"

        mock_api_class.return_value.list_spaces.return_value = iter([mock_space])

        results = list(iter_spaces())

        assert len(results) == 1
        assert isinstance(results[0], SpaceInfo)

    @patch("huggingface_hub.HfApi")
    def test_passes_sdk_filter_to_api(self, mock_api_class: MagicMock) -> None:
        """Test that SDK filter is passed to HfApi."""
        mock_api_class.return_value.list_spaces.return_value = iter([])

        list(iter_spaces(sdk="gradio"))

        mock_api_class.return_value.list_spaces.assert_called_once_with(
            author=None,
            filter="gradio",
        )


class TestGetSpaceInfo:
    """Tests for get_space_info function."""

    def test_empty_space_id_raises_error(self) -> None:
        """Test that empty space_id raises ValueError."""
        with pytest.raises(ValueError, match="space_id cannot be empty"):
            get_space_info("")

    @patch("huggingface_hub.HfApi")
    def test_returns_space_info(self, mock_api_class: MagicMock) -> None:
        """Test that get_space_info returns SpaceInfo."""
        mock_space = MagicMock()
        mock_space.id = "gradio/hello-world"
        mock_space.author = "gradio"
        mock_space.sdk = "gradio"
        mock_space.likes = 1000
        mock_space.title = "Hello World"
        mock_space.emoji = "🌎"
        mock_space.hardware = "cpu-basic"

        mock_api_class.return_value.space_info.return_value = mock_space

        result = get_space_info("gradio/hello-world")

        assert isinstance(result, SpaceInfo)
        assert result.space_id == "gradio/hello-world"
        mock_api_class.return_value.space_info.assert_called_once_with(
            "gradio/hello-world"
        )


class TestGetSpaceRuntime:
    """Tests for get_space_runtime function."""

    def test_empty_space_id_raises_error(self) -> None:
        """Test that empty space_id raises ValueError."""
        with pytest.raises(ValueError, match="space_id cannot be empty"):
            get_space_runtime("")

    @patch("huggingface_hub.HfApi")
    def test_returns_space_runtime(self, mock_api_class: MagicMock) -> None:
        """Test that get_space_runtime returns SpaceRuntime."""
        mock_runtime = MagicMock()
        mock_runtime.stage = "RUNNING"
        mock_runtime.hardware = "cpu-basic"
        mock_runtime.sdk_version = "3.50.0"

        mock_api_class.return_value.get_space_runtime.return_value = mock_runtime

        result = get_space_runtime("gradio/hello-world")

        assert isinstance(result, SpaceRuntime)
        assert result.stage == "RUNNING"
        assert result.hardware == "cpu-basic"
        assert result.sdk_version == "3.50.0"

    @patch("huggingface_hub.HfApi")
    def test_handles_missing_sdk_version(self, mock_api_class: MagicMock) -> None:
        """Test handling missing sdk_version attribute."""
        mock_runtime = MagicMock(spec=["stage", "hardware"])
        mock_runtime.stage = "RUNNING"
        mock_runtime.hardware = "cpu-basic"

        mock_api_class.return_value.get_space_runtime.return_value = mock_runtime

        result = get_space_runtime("test/space")

        assert result.sdk_version is None


class TestListSdks:
    """Tests for list_sdks function."""

    def test_returns_list(self) -> None:
        """Test that list_sdks returns a list."""
        result = list_sdks()
        assert isinstance(result, list)

    def test_contains_expected_sdks(self) -> None:
        """Test that list contains expected SDKs."""
        result = list_sdks()
        assert "gradio" in result
        assert "streamlit" in result
        assert "docker" in result
        assert "static" in result

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_sdks()
        assert result == sorted(result)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        result = list_sdks()
        assert all(isinstance(s, str) for s in result)


class TestListHardwareTiers:
    """Tests for list_hardware_tiers function."""

    def test_returns_list(self) -> None:
        """Test that list_hardware_tiers returns a list."""
        result = list_hardware_tiers()
        assert isinstance(result, list)

    def test_contains_expected_hardware(self) -> None:
        """Test that list contains expected hardware tiers."""
        result = list_hardware_tiers()
        assert "cpu-basic" in result
        assert "t4-small" in result
        assert "a10g-small" in result

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_hardware_tiers()
        assert result == sorted(result)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        result = list_hardware_tiers()
        assert all(isinstance(h, str) for h in result)


class TestValidateSdk:
    """Tests for validate_sdk function."""

    def test_valid_gradio(self) -> None:
        """Test validation of gradio SDK."""
        assert validate_sdk("gradio") is True

    def test_valid_streamlit(self) -> None:
        """Test validation of streamlit SDK."""
        assert validate_sdk("streamlit") is True

    def test_valid_docker(self) -> None:
        """Test validation of docker SDK."""
        assert validate_sdk("docker") is True

    def test_valid_static(self) -> None:
        """Test validation of static SDK."""
        assert validate_sdk("static") is True

    def test_invalid_sdk(self) -> None:
        """Test validation of invalid SDK."""
        assert validate_sdk("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_sdk("") is False

    def test_case_sensitive(self) -> None:
        """Test that validation is case sensitive."""
        assert validate_sdk("Gradio") is False
        assert validate_sdk("GRADIO") is False


class TestValidateHardware:
    """Tests for validate_hardware function."""

    def test_valid_cpu_basic(self) -> None:
        """Test validation of cpu-basic hardware."""
        assert validate_hardware("cpu-basic") is True

    def test_valid_t4_small(self) -> None:
        """Test validation of t4-small hardware."""
        assert validate_hardware("t4-small") is True

    def test_valid_a10g_small(self) -> None:
        """Test validation of a10g-small hardware."""
        assert validate_hardware("a10g-small") is True

    def test_invalid_hardware(self) -> None:
        """Test validation of invalid hardware."""
        assert validate_hardware("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_hardware("") is False

    def test_case_sensitive(self) -> None:
        """Test that validation is case sensitive."""
        assert validate_hardware("CPU-BASIC") is False
        assert validate_hardware("Cpu-Basic") is False
