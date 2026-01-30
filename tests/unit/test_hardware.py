"""Tests for inference.hardware module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hf_gtc.inference.hardware import (
    VALID_COMPUTE_CAPABILITIES,
    VALID_DEVICE_TYPES,
    VALID_GPU_ARCHITECTURES,
    ComputeCapability,
    DeviceConfig,
    DeviceMap,
    DeviceType,
    GPUArchitecture,
    GPUInfo,
    HardwareStats,
    calculate_memory_bandwidth,
    check_hardware_compatibility,
    create_device_config,
    create_device_map,
    create_gpu_info,
    create_hardware_stats,
    detect_available_devices,
    estimate_throughput,
    format_hardware_stats,
    get_architecture_for_compute_capability,
    get_compute_capability,
    get_device_type,
    get_gpu_architecture,
    get_optimal_device_map,
    get_recommended_hardware_config,
    list_compute_capabilities,
    list_device_types,
    list_gpu_architectures,
    validate_device_config,
    validate_device_map,
    validate_gpu_info,
    validate_hardware_stats,
)


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for device_type in DeviceType:
            assert isinstance(device_type.value, str)

    def test_cpu_value(self) -> None:
        """CPU has correct value."""
        assert DeviceType.CPU.value == "cpu"

    def test_cuda_value(self) -> None:
        """CUDA has correct value."""
        assert DeviceType.CUDA.value == "cuda"

    def test_mps_value(self) -> None:
        """MPS has correct value."""
        assert DeviceType.MPS.value == "mps"

    def test_tpu_value(self) -> None:
        """TPU has correct value."""
        assert DeviceType.TPU.value == "tpu"

    def test_xpu_value(self) -> None:
        """XPU has correct value."""
        assert DeviceType.XPU.value == "xpu"

    def test_valid_device_types_frozenset(self) -> None:
        """VALID_DEVICE_TYPES is a frozenset."""
        assert isinstance(VALID_DEVICE_TYPES, frozenset)

    def test_valid_device_types_count(self) -> None:
        """VALID_DEVICE_TYPES has correct count."""
        assert len(VALID_DEVICE_TYPES) == 5


class TestGPUArchitecture:
    """Tests for GPUArchitecture enum."""

    def test_all_archs_have_values(self) -> None:
        """All architectures have string values."""
        for arch in GPUArchitecture:
            assert isinstance(arch.value, str)

    def test_volta_value(self) -> None:
        """VOLTA has correct value."""
        assert GPUArchitecture.VOLTA.value == "volta"

    def test_turing_value(self) -> None:
        """TURING has correct value."""
        assert GPUArchitecture.TURING.value == "turing"

    def test_ampere_value(self) -> None:
        """AMPERE has correct value."""
        assert GPUArchitecture.AMPERE.value == "ampere"

    def test_hopper_value(self) -> None:
        """HOPPER has correct value."""
        assert GPUArchitecture.HOPPER.value == "hopper"

    def test_ada_lovelace_value(self) -> None:
        """ADA_LOVELACE has correct value."""
        assert GPUArchitecture.ADA_LOVELACE.value == "ada_lovelace"

    def test_valid_gpu_architectures_frozenset(self) -> None:
        """VALID_GPU_ARCHITECTURES is a frozenset."""
        assert isinstance(VALID_GPU_ARCHITECTURES, frozenset)

    def test_valid_gpu_architectures_count(self) -> None:
        """VALID_GPU_ARCHITECTURES has correct count."""
        assert len(VALID_GPU_ARCHITECTURES) == 5


class TestComputeCapability:
    """Tests for ComputeCapability enum."""

    def test_all_caps_have_values(self) -> None:
        """All compute capabilities have string values."""
        for cap in ComputeCapability:
            assert isinstance(cap.value, str)

    def test_sm_70_value(self) -> None:
        """SM_70 has correct value."""
        assert ComputeCapability.SM_70.value == "sm_70"

    def test_sm_75_value(self) -> None:
        """SM_75 has correct value."""
        assert ComputeCapability.SM_75.value == "sm_75"

    def test_sm_80_value(self) -> None:
        """SM_80 has correct value."""
        assert ComputeCapability.SM_80.value == "sm_80"

    def test_sm_86_value(self) -> None:
        """SM_86 has correct value."""
        assert ComputeCapability.SM_86.value == "sm_86"

    def test_sm_89_value(self) -> None:
        """SM_89 has correct value."""
        assert ComputeCapability.SM_89.value == "sm_89"

    def test_sm_90_value(self) -> None:
        """SM_90 has correct value."""
        assert ComputeCapability.SM_90.value == "sm_90"

    def test_valid_compute_capabilities_frozenset(self) -> None:
        """VALID_COMPUTE_CAPABILITIES is a frozenset."""
        assert isinstance(VALID_COMPUTE_CAPABILITIES, frozenset)

    def test_valid_compute_capabilities_count(self) -> None:
        """VALID_COMPUTE_CAPABILITIES has correct count."""
        assert len(VALID_COMPUTE_CAPABILITIES) == 6


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_create_gpu_info(self) -> None:
        """Create GPU info."""
        info = GPUInfo(
            name="A100",
            memory_gb=80.0,
            compute_capability=ComputeCapability.SM_80,
            tensor_cores=True,
        )
        assert info.name == "A100"
        assert info.memory_gb == 80.0

    def test_gpu_info_is_frozen(self) -> None:
        """GPU info is immutable."""
        info = GPUInfo("A100", 80.0, ComputeCapability.SM_80, True)
        with pytest.raises(AttributeError):
            info.name = "V100"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        info = GPUInfo("RTX 4090", 24.0, ComputeCapability.SM_89, True)
        assert info.name == "RTX 4090"
        assert info.memory_gb == 24.0
        assert info.compute_capability == ComputeCapability.SM_89
        assert info.tensor_cores is True


class TestDeviceConfig:
    """Tests for DeviceConfig dataclass."""

    def test_create_device_config(self) -> None:
        """Create device config."""
        config = DeviceConfig(
            device_type=DeviceType.CUDA,
            device_ids=(0, 1),
            memory_fraction=0.9,
            allow_growth=True,
        )
        assert config.device_type == DeviceType.CUDA
        assert config.device_ids == (0, 1)

    def test_device_config_is_frozen(self) -> None:
        """Device config is immutable."""
        config = DeviceConfig(DeviceType.CUDA, (0,), 0.9, True)
        with pytest.raises(AttributeError):
            config.memory_fraction = 0.8  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        config = DeviceConfig(DeviceType.MPS, (0,), 0.8, False)
        assert config.device_type == DeviceType.MPS
        assert config.device_ids == (0,)
        assert config.memory_fraction == 0.8
        assert config.allow_growth is False


class TestDeviceMap:
    """Tests for DeviceMap dataclass."""

    def test_create_device_map(self) -> None:
        """Create device map."""
        dm = DeviceMap(
            layer_mapping={"layer_0": 0, "layer_1": 1},
            offload_folder="/tmp/offload",
            offload_buffers=True,
        )
        assert dm.layer_mapping["layer_0"] == 0
        assert dm.offload_folder == "/tmp/offload"

    def test_device_map_is_frozen(self) -> None:
        """Device map is immutable."""
        dm = DeviceMap({"layer_0": 0}, None, False)
        with pytest.raises(AttributeError):
            dm.offload_folder = "/new/path"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        dm = DeviceMap({"layer_0": 0}, "/tmp", True)
        assert dm.layer_mapping == {"layer_0": 0}
        assert dm.offload_folder == "/tmp"
        assert dm.offload_buffers is True


class TestHardwareStats:
    """Tests for HardwareStats dataclass."""

    def test_create_hardware_stats(self) -> None:
        """Create hardware stats."""
        stats = HardwareStats(
            device_count=8,
            total_memory_gb=640.0,
            compute_tflops=156.0,
            memory_bandwidth_gbps=2039.0,
        )
        assert stats.device_count == 8
        assert stats.total_memory_gb == 640.0

    def test_hardware_stats_is_frozen(self) -> None:
        """Hardware stats is immutable."""
        stats = HardwareStats(8, 640.0, 156.0, 2039.0)
        with pytest.raises(AttributeError):
            stats.device_count = 16  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        stats = HardwareStats(4, 320.0, 78.0, 1020.0)
        assert stats.device_count == 4
        assert stats.total_memory_gb == 320.0
        assert stats.compute_tflops == 78.0
        assert stats.memory_bandwidth_gbps == 1020.0


class TestValidateGPUInfo:
    """Tests for validate_gpu_info function."""

    def test_valid_gpu_info(self) -> None:
        """Valid GPU info passes validation."""
        info = GPUInfo("A100", 80.0, ComputeCapability.SM_80, True)
        validate_gpu_info(info)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        info = GPUInfo("", 80.0, ComputeCapability.SM_80, True)
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_gpu_info(info)

    def test_zero_memory_raises(self) -> None:
        """Zero memory raises ValueError."""
        info = GPUInfo("A100", 0.0, ComputeCapability.SM_80, True)
        with pytest.raises(ValueError, match="memory_gb must be positive"):
            validate_gpu_info(info)

    def test_negative_memory_raises(self) -> None:
        """Negative memory raises ValueError."""
        info = GPUInfo("A100", -1.0, ComputeCapability.SM_80, True)
        with pytest.raises(ValueError, match="memory_gb must be positive"):
            validate_gpu_info(info)


class TestValidateDeviceConfig:
    """Tests for validate_device_config function."""

    def test_valid_device_config(self) -> None:
        """Valid device config passes validation."""
        config = DeviceConfig(DeviceType.CUDA, (0,), 0.9, True)
        validate_device_config(config)

    def test_memory_fraction_too_high_raises(self) -> None:
        """Memory fraction > 1 raises ValueError."""
        config = DeviceConfig(DeviceType.CUDA, (0,), 1.5, True)
        with pytest.raises(ValueError, match="memory_fraction must be between"):
            validate_device_config(config)

    def test_memory_fraction_zero_raises(self) -> None:
        """Memory fraction = 0 raises ValueError."""
        config = DeviceConfig(DeviceType.CUDA, (0,), 0.0, True)
        with pytest.raises(ValueError, match="memory_fraction must be between"):
            validate_device_config(config)

    def test_memory_fraction_negative_raises(self) -> None:
        """Negative memory fraction raises ValueError."""
        config = DeviceConfig(DeviceType.CUDA, (0,), -0.5, True)
        with pytest.raises(ValueError, match="memory_fraction must be between"):
            validate_device_config(config)

    def test_empty_device_ids_raises(self) -> None:
        """Empty device IDs raises ValueError."""
        config = DeviceConfig(DeviceType.CUDA, (), 0.9, True)
        with pytest.raises(ValueError, match="device_ids cannot be empty"):
            validate_device_config(config)

    def test_negative_device_id_raises(self) -> None:
        """Negative device ID raises ValueError."""
        config = DeviceConfig(DeviceType.CUDA, (-1,), 0.9, True)
        with pytest.raises(ValueError, match="device_ids must be non-negative"):
            validate_device_config(config)


class TestValidateDeviceMap:
    """Tests for validate_device_map function."""

    def test_valid_device_map(self) -> None:
        """Valid device map passes validation."""
        dm = DeviceMap({"layer_0": 0}, None, False)
        validate_device_map(dm)

    def test_empty_layer_mapping_raises(self) -> None:
        """Empty layer mapping raises ValueError."""
        dm = DeviceMap({}, None, False)
        with pytest.raises(ValueError, match="layer_mapping cannot be empty"):
            validate_device_map(dm)

    def test_empty_layer_name_raises(self) -> None:
        """Empty layer name raises ValueError."""
        dm = DeviceMap({"": 0}, None, False)
        with pytest.raises(ValueError, match="layer names cannot be empty"):
            validate_device_map(dm)

    def test_negative_device_id_raises(self) -> None:
        """Negative device ID raises ValueError."""
        dm = DeviceMap({"layer_0": -2}, None, False)
        with pytest.raises(ValueError, match="device_id must be non-negative"):
            validate_device_map(dm)


class TestValidateHardwareStats:
    """Tests for validate_hardware_stats function."""

    def test_valid_hardware_stats(self) -> None:
        """Valid hardware stats passes validation."""
        stats = HardwareStats(8, 640.0, 156.0, 2039.0)
        validate_hardware_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        stats = HardwareStats(0, 0.0, 0.0, 0.0)
        validate_hardware_stats(stats)

    def test_negative_device_count_raises(self) -> None:
        """Negative device count raises ValueError."""
        stats = HardwareStats(-1, 640.0, 156.0, 2039.0)
        with pytest.raises(ValueError, match="device_count cannot be negative"):
            validate_hardware_stats(stats)

    def test_negative_memory_raises(self) -> None:
        """Negative memory raises ValueError."""
        stats = HardwareStats(8, -1.0, 156.0, 2039.0)
        with pytest.raises(ValueError, match="total_memory_gb cannot be negative"):
            validate_hardware_stats(stats)

    def test_negative_compute_raises(self) -> None:
        """Negative compute raises ValueError."""
        stats = HardwareStats(8, 640.0, -1.0, 2039.0)
        with pytest.raises(ValueError, match="compute_tflops cannot be negative"):
            validate_hardware_stats(stats)

    def test_negative_bandwidth_raises(self) -> None:
        """Negative bandwidth raises ValueError."""
        stats = HardwareStats(8, 640.0, 156.0, -1.0)
        with pytest.raises(ValueError, match="memory_bandwidth_gbps cannot be"):
            validate_hardware_stats(stats)


class TestCreateGPUInfo:
    """Tests for create_gpu_info function."""

    def test_default_gpu_info(self) -> None:
        """Create GPU info with defaults."""
        info = create_gpu_info("A100", 80.0)
        assert info.name == "A100"
        assert info.memory_gb == 80.0
        assert info.compute_capability == ComputeCapability.SM_80
        assert info.tensor_cores is True

    def test_custom_compute_capability(self) -> None:
        """Create GPU info with custom compute capability."""
        info = create_gpu_info("RTX 4090", 24.0, "sm_89")
        assert info.compute_capability == ComputeCapability.SM_89

    def test_custom_tensor_cores(self) -> None:
        """Create GPU info with custom tensor cores."""
        info = create_gpu_info("Old GPU", 8.0, "sm_70", tensor_cores=False)
        assert info.tensor_cores is False

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_gpu_info("", 80.0)

    def test_invalid_compute_capability_raises(self) -> None:
        """Invalid compute capability raises ValueError."""
        with pytest.raises(ValueError, match="compute_capability must be one of"):
            create_gpu_info("A100", 80.0, "invalid")  # type: ignore[arg-type]


class TestCreateDeviceConfig:
    """Tests for create_device_config function."""

    def test_default_device_config(self) -> None:
        """Create device config with defaults."""
        config = create_device_config()
        assert config.device_type == DeviceType.CUDA
        assert config.device_ids == (0,)
        assert config.memory_fraction == 0.9
        assert config.allow_growth is True

    def test_custom_device_type(self) -> None:
        """Create device config with custom device type."""
        config = create_device_config("cpu")
        assert config.device_type == DeviceType.CPU

    def test_custom_device_ids(self) -> None:
        """Create device config with custom device IDs."""
        config = create_device_config(device_ids=(0, 1, 2))
        assert config.device_ids == (0, 1, 2)

    def test_custom_memory_fraction(self) -> None:
        """Create device config with custom memory fraction."""
        config = create_device_config(memory_fraction=0.8)
        assert config.memory_fraction == 0.8

    def test_custom_allow_growth(self) -> None:
        """Create device config with custom allow growth."""
        config = create_device_config(allow_growth=False)
        assert config.allow_growth is False

    def test_invalid_device_type_raises(self) -> None:
        """Invalid device type raises ValueError."""
        with pytest.raises(ValueError, match="device_type must be one of"):
            create_device_config("invalid")  # type: ignore[arg-type]

    def test_invalid_memory_fraction_raises(self) -> None:
        """Invalid memory fraction raises ValueError."""
        with pytest.raises(ValueError, match="memory_fraction must be between"):
            create_device_config(memory_fraction=1.5)


class TestCreateDeviceMap:
    """Tests for create_device_map function."""

    def test_basic_device_map(self) -> None:
        """Create basic device map."""
        dm = create_device_map({"layer_0": 0, "layer_1": 1})
        assert dm.layer_mapping["layer_0"] == 0
        assert dm.layer_mapping["layer_1"] == 1
        assert dm.offload_folder is None
        assert dm.offload_buffers is False

    def test_with_offload(self) -> None:
        """Create device map with offloading."""
        dm = create_device_map({"layer_0": 0}, "/tmp/offload", True)
        assert dm.offload_folder == "/tmp/offload"
        assert dm.offload_buffers is True

    def test_empty_mapping_raises(self) -> None:
        """Empty mapping raises ValueError."""
        with pytest.raises(ValueError, match="layer_mapping cannot be empty"):
            create_device_map({})


class TestCreateHardwareStats:
    """Tests for create_hardware_stats function."""

    def test_default_hardware_stats(self) -> None:
        """Create hardware stats with defaults."""
        stats = create_hardware_stats()
        assert stats.device_count == 1
        assert stats.total_memory_gb == 0.0
        assert stats.compute_tflops == 0.0
        assert stats.memory_bandwidth_gbps == 0.0

    def test_custom_values(self) -> None:
        """Create hardware stats with custom values."""
        stats = create_hardware_stats(8, 640.0, 156.0, 2039.0)
        assert stats.device_count == 8
        assert stats.total_memory_gb == 640.0
        assert stats.compute_tflops == 156.0
        assert stats.memory_bandwidth_gbps == 2039.0

    def test_negative_device_count_raises(self) -> None:
        """Negative device count raises ValueError."""
        with pytest.raises(ValueError, match="device_count cannot be negative"):
            create_hardware_stats(-1)


class TestListDeviceTypes:
    """Tests for list_device_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_device_types()
        assert types == sorted(types)

    def test_contains_cpu(self) -> None:
        """Contains cpu."""
        types = list_device_types()
        assert "cpu" in types

    def test_contains_cuda(self) -> None:
        """Contains cuda."""
        types = list_device_types()
        assert "cuda" in types

    def test_contains_all_types(self) -> None:
        """Contains all types."""
        types = list_device_types()
        assert len(types) == len(VALID_DEVICE_TYPES)


class TestListGPUArchitectures:
    """Tests for list_gpu_architectures function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        archs = list_gpu_architectures()
        assert archs == sorted(archs)

    def test_contains_ampere(self) -> None:
        """Contains ampere."""
        archs = list_gpu_architectures()
        assert "ampere" in archs

    def test_contains_hopper(self) -> None:
        """Contains hopper."""
        archs = list_gpu_architectures()
        assert "hopper" in archs

    def test_contains_all_archs(self) -> None:
        """Contains all architectures."""
        archs = list_gpu_architectures()
        assert len(archs) == len(VALID_GPU_ARCHITECTURES)


class TestListComputeCapabilities:
    """Tests for list_compute_capabilities function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        caps = list_compute_capabilities()
        assert caps == sorted(caps)

    def test_contains_sm_80(self) -> None:
        """Contains sm_80."""
        caps = list_compute_capabilities()
        assert "sm_80" in caps

    def test_contains_sm_90(self) -> None:
        """Contains sm_90."""
        caps = list_compute_capabilities()
        assert "sm_90" in caps

    def test_contains_all_caps(self) -> None:
        """Contains all compute capabilities."""
        caps = list_compute_capabilities()
        assert len(caps) == len(VALID_COMPUTE_CAPABILITIES)


class TestGetDeviceType:
    """Tests for get_device_type function."""

    def test_get_cpu(self) -> None:
        """Get cpu device type."""
        device = get_device_type("cpu")
        assert device == DeviceType.CPU

    def test_get_cuda(self) -> None:
        """Get cuda device type."""
        device = get_device_type("cuda")
        assert device == DeviceType.CUDA

    def test_get_mps(self) -> None:
        """Get mps device type."""
        device = get_device_type("mps")
        assert device == DeviceType.MPS

    def test_get_tpu(self) -> None:
        """Get tpu device type."""
        device = get_device_type("tpu")
        assert device == DeviceType.TPU

    def test_get_xpu(self) -> None:
        """Get xpu device type."""
        device = get_device_type("xpu")
        assert device == DeviceType.XPU

    def test_invalid_device_type_raises(self) -> None:
        """Invalid device type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device type"):
            get_device_type("invalid")


class TestGetGPUArchitecture:
    """Tests for get_gpu_architecture function."""

    def test_get_volta(self) -> None:
        """Get volta architecture."""
        arch = get_gpu_architecture("volta")
        assert arch == GPUArchitecture.VOLTA

    def test_get_turing(self) -> None:
        """Get turing architecture."""
        arch = get_gpu_architecture("turing")
        assert arch == GPUArchitecture.TURING

    def test_get_ampere(self) -> None:
        """Get ampere architecture."""
        arch = get_gpu_architecture("ampere")
        assert arch == GPUArchitecture.AMPERE

    def test_get_hopper(self) -> None:
        """Get hopper architecture."""
        arch = get_gpu_architecture("hopper")
        assert arch == GPUArchitecture.HOPPER

    def test_get_ada_lovelace(self) -> None:
        """Get ada_lovelace architecture."""
        arch = get_gpu_architecture("ada_lovelace")
        assert arch == GPUArchitecture.ADA_LOVELACE

    def test_invalid_architecture_raises(self) -> None:
        """Invalid architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unknown GPU architecture"):
            get_gpu_architecture("invalid")


class TestGetComputeCapability:
    """Tests for get_compute_capability function."""

    def test_get_sm_70(self) -> None:
        """Get sm_70 compute capability."""
        cap = get_compute_capability("sm_70")
        assert cap == ComputeCapability.SM_70

    def test_get_sm_80(self) -> None:
        """Get sm_80 compute capability."""
        cap = get_compute_capability("sm_80")
        assert cap == ComputeCapability.SM_80

    def test_get_sm_90(self) -> None:
        """Get sm_90 compute capability."""
        cap = get_compute_capability("sm_90")
        assert cap == ComputeCapability.SM_90

    def test_invalid_capability_raises(self) -> None:
        """Invalid capability raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compute capability"):
            get_compute_capability("invalid")


class TestDetectAvailableDevices:
    """Tests for detect_available_devices function."""

    def test_cpu_always_available(self) -> None:
        """CPU is always available."""
        devices = detect_available_devices()
        assert "cpu" in devices

    def test_returns_list(self) -> None:
        """Returns a list."""
        devices = detect_available_devices()
        assert isinstance(devices, list)

    def test_cuda_available(self) -> None:
        """Test CUDA detection when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Re-import to use mocked module
            from hf_gtc.inference import hardware

            devices = hardware.detect_available_devices()
            assert "cpu" in devices

    def test_no_gpu_available(self) -> None:
        """Test when no GPU available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch.dict("sys.modules", {"jax": None}),
        ):
            devices = detect_available_devices()
            assert "cpu" in devices


class TestGetOptimalDeviceMap:
    """Tests for get_optimal_device_map function."""

    def test_basic_distribution(self) -> None:
        """Basic layer distribution."""
        dm = get_optimal_device_map(4, 1.0, gpu_count=2, gpu_memory_gb=10.0)
        assert len(dm.layer_mapping) == 4

    def test_single_gpu(self) -> None:
        """Single GPU distribution."""
        dm = get_optimal_device_map(10, 0.5, gpu_count=1, gpu_memory_gb=24.0)
        assert len(dm.layer_mapping) == 10
        # All layers should be on GPU 0
        for device_id in dm.layer_mapping.values():
            assert device_id in (0, -1)  # 0 for GPU, -1 for CPU offload

    def test_cpu_offload_needed(self) -> None:
        """Test when CPU offload is needed."""
        dm = get_optimal_device_map(100, 1.0, gpu_count=1, gpu_memory_gb=10.0)
        assert dm.offload_folder is not None
        assert dm.offload_buffers is True
        # Some layers should be offloaded to CPU
        assert -1 in dm.layer_mapping.values()

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="model_layers must be positive"):
            get_optimal_device_map(0, 1.0)

    def test_negative_layers_raises(self) -> None:
        """Negative layers raises ValueError."""
        with pytest.raises(ValueError, match="model_layers must be positive"):
            get_optimal_device_map(-1, 1.0)

    def test_zero_layer_memory_raises(self) -> None:
        """Zero layer memory raises ValueError."""
        with pytest.raises(ValueError, match="layer_memory_gb must be positive"):
            get_optimal_device_map(10, 0.0)

    def test_zero_gpu_count_raises(self) -> None:
        """Zero GPU count raises ValueError."""
        with pytest.raises(ValueError, match="gpu_count must be positive"):
            get_optimal_device_map(10, 1.0, gpu_count=0)

    def test_zero_gpu_memory_raises(self) -> None:
        """Zero GPU memory raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_gb must be positive"):
            get_optimal_device_map(10, 1.0, gpu_memory_gb=0.0)


class TestEstimateThroughput:
    """Tests for estimate_throughput function."""

    def test_basic_calculation(self) -> None:
        """Basic throughput calculation."""
        throughput = estimate_throughput("cuda", 32, 512)
        assert throughput > 0

    def test_cuda_faster_than_cpu(self) -> None:
        """CUDA faster than CPU."""
        cpu_tp = estimate_throughput("cpu", 32, 512)
        cuda_tp = estimate_throughput("cuda", 32, 512)
        assert cuda_tp > cpu_tp

    def test_larger_batch_higher_throughput(self) -> None:
        """Larger batch gives higher throughput."""
        tp_small = estimate_throughput("cuda", 16, 512)
        tp_large = estimate_throughput("cuda", 32, 512)
        assert tp_large > tp_small

    def test_all_device_types(self) -> None:
        """All device types work."""
        for device_type in VALID_DEVICE_TYPES:
            throughput = estimate_throughput(device_type, 32, 512)  # type: ignore[arg-type]
            assert throughput > 0

    def test_invalid_device_type_raises(self) -> None:
        """Invalid device type raises ValueError."""
        with pytest.raises(ValueError, match="device_type must be one of"):
            estimate_throughput("invalid", 32, 512)  # type: ignore[arg-type]

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_throughput("cuda", 0, 512)

    def test_zero_sequence_length_raises(self) -> None:
        """Zero sequence length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_throughput("cuda", 32, 0)

    def test_zero_hidden_size_raises(self) -> None:
        """Zero hidden size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_throughput("cuda", 32, 512, hidden_size=0)


class TestCalculateMemoryBandwidth:
    """Tests for calculate_memory_bandwidth function."""

    def test_cpu_bandwidth(self) -> None:
        """CPU bandwidth."""
        bw = calculate_memory_bandwidth("cpu")
        assert bw > 0
        assert bw < 100  # CPU memory is slower

    def test_cuda_bandwidth_default(self) -> None:
        """CUDA bandwidth with default."""
        bw = calculate_memory_bandwidth("cuda")
        assert bw > 0

    def test_cuda_bandwidth_with_capability(self) -> None:
        """CUDA bandwidth with compute capability."""
        bw = calculate_memory_bandwidth("cuda", "sm_80")
        assert bw == 2039.0  # A100

    def test_h100_bandwidth(self) -> None:
        """H100 has highest bandwidth."""
        bw_h100 = calculate_memory_bandwidth("cuda", "sm_90")
        bw_a100 = calculate_memory_bandwidth("cuda", "sm_80")
        assert bw_h100 > bw_a100

    def test_all_compute_capabilities(self) -> None:
        """All compute capabilities work."""
        for cap in VALID_COMPUTE_CAPABILITIES:
            bw = calculate_memory_bandwidth("cuda", cap)  # type: ignore[arg-type]
            assert bw > 0

    def test_invalid_device_type_raises(self) -> None:
        """Invalid device type raises ValueError."""
        with pytest.raises(ValueError, match="device_type must be one of"):
            calculate_memory_bandwidth("invalid")  # type: ignore[arg-type]

    def test_invalid_compute_capability_raises(self) -> None:
        """Invalid compute capability raises ValueError."""
        with pytest.raises(ValueError, match="compute_capability must be one of"):
            calculate_memory_bandwidth("cuda", "invalid")  # type: ignore[arg-type]


class TestCheckHardwareCompatibility:
    """Tests for check_hardware_compatibility function."""

    def test_compatible_hardware(self) -> None:
        """Compatible hardware returns True."""
        compat, msg = check_hardware_compatibility(10.0, 50.0, 24.0, 100.0)
        assert compat is True
        assert "compatible" in msg.lower()

    def test_insufficient_memory(self) -> None:
        """Insufficient memory returns False."""
        compat, msg = check_hardware_compatibility(50.0, 50.0, 24.0, 100.0)
        assert compat is False
        assert "memory" in msg.lower()

    def test_insufficient_compute(self) -> None:
        """Insufficient compute returns False."""
        compat, msg = check_hardware_compatibility(10.0, 200.0, 24.0, 100.0)
        assert compat is False
        assert "compute" in msg.lower()

    def test_zero_model_memory_raises(self) -> None:
        """Zero model memory raises ValueError."""
        with pytest.raises(ValueError, match="model_memory_gb must be positive"):
            check_hardware_compatibility(0.0, 50.0, 24.0, 100.0)

    def test_negative_model_memory_raises(self) -> None:
        """Negative model memory raises ValueError."""
        with pytest.raises(ValueError, match="model_memory_gb must be positive"):
            check_hardware_compatibility(-1.0, 50.0, 24.0, 100.0)

    def test_zero_model_compute_raises(self) -> None:
        """Zero model compute raises ValueError."""
        with pytest.raises(ValueError, match="model_compute_tflops must be positive"):
            check_hardware_compatibility(10.0, 0.0, 24.0, 100.0)

    def test_zero_device_memory_raises(self) -> None:
        """Zero device memory raises ValueError."""
        with pytest.raises(ValueError, match="device_memory_gb must be positive"):
            check_hardware_compatibility(10.0, 50.0, 0.0, 100.0)

    def test_zero_device_compute_raises(self) -> None:
        """Zero device compute raises ValueError."""
        with pytest.raises(ValueError, match="device_compute_tflops must be positive"):
            check_hardware_compatibility(10.0, 50.0, 24.0, 0.0)


class TestFormatHardwareStats:
    """Tests for format_hardware_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = HardwareStats(8, 640.0, 156.0, 2039.0)
        formatted = format_hardware_stats(stats)
        assert "Device Count: 8" in formatted
        assert "Total Memory: 640.00 GB" in formatted
        assert "Compute: 156.00 TFLOPS" in formatted
        assert "Memory Bandwidth: 2039.00 GB/s" in formatted

    def test_zero_values(self) -> None:
        """Zero values format correctly."""
        stats = HardwareStats(0, 0.0, 0.0, 0.0)
        formatted = format_hardware_stats(stats)
        assert "Device Count: 0" in formatted
        assert "Total Memory: 0.00 GB" in formatted

    def test_multiline_output(self) -> None:
        """Output contains multiple lines."""
        stats = HardwareStats(8, 640.0, 156.0, 2039.0)
        formatted = format_hardware_stats(stats)
        lines = formatted.split("\n")
        assert len(lines) == 4


class TestGetRecommendedHardwareConfig:
    """Tests for get_recommended_hardware_config function."""

    def test_cpu_config(self) -> None:
        """CPU config."""
        config = get_recommended_hardware_config("cpu")
        assert config.device_type == DeviceType.CPU
        assert config.memory_fraction == 1.0

    def test_gpu_consumer_config(self) -> None:
        """GPU consumer config."""
        config = get_recommended_hardware_config("gpu_consumer")
        assert config.device_type == DeviceType.CUDA
        assert config.memory_fraction == 0.8
        assert config.allow_growth is True

    def test_gpu_datacenter_config(self) -> None:
        """GPU datacenter config."""
        config = get_recommended_hardware_config("gpu_datacenter")
        assert config.device_type == DeviceType.CUDA
        assert config.memory_fraction == 0.9
        assert config.allow_growth is False

    def test_tpu_config(self) -> None:
        """TPU config."""
        config = get_recommended_hardware_config("tpu")
        assert config.device_type == DeviceType.TPU
        assert config.memory_fraction == 0.95

    def test_edge_config(self) -> None:
        """Edge config."""
        config = get_recommended_hardware_config("edge")
        assert config.device_type == DeviceType.CPU
        assert config.memory_fraction == 0.7

    def test_default_is_datacenter(self) -> None:
        """Default is gpu_datacenter."""
        config = get_recommended_hardware_config()
        assert config.device_type == DeviceType.CUDA

    def test_invalid_target_raises(self) -> None:
        """Invalid target raises ValueError."""
        with pytest.raises(ValueError, match="Unknown target"):
            get_recommended_hardware_config("invalid")  # type: ignore[arg-type]


class TestGetArchitectureForComputeCapability:
    """Tests for get_architecture_for_compute_capability function."""

    def test_sm_70_volta(self) -> None:
        """SM_70 is Volta."""
        arch = get_architecture_for_compute_capability("sm_70")
        assert arch == GPUArchitecture.VOLTA

    def test_sm_75_turing(self) -> None:
        """SM_75 is Turing."""
        arch = get_architecture_for_compute_capability("sm_75")
        assert arch == GPUArchitecture.TURING

    def test_sm_80_ampere(self) -> None:
        """SM_80 is Ampere."""
        arch = get_architecture_for_compute_capability("sm_80")
        assert arch == GPUArchitecture.AMPERE

    def test_sm_86_ampere(self) -> None:
        """SM_86 is Ampere."""
        arch = get_architecture_for_compute_capability("sm_86")
        assert arch == GPUArchitecture.AMPERE

    def test_sm_89_ada_lovelace(self) -> None:
        """SM_89 is Ada Lovelace."""
        arch = get_architecture_for_compute_capability("sm_89")
        assert arch == GPUArchitecture.ADA_LOVELACE

    def test_sm_90_hopper(self) -> None:
        """SM_90 is Hopper."""
        arch = get_architecture_for_compute_capability("sm_90")
        assert arch == GPUArchitecture.HOPPER

    def test_invalid_capability_raises(self) -> None:
        """Invalid capability raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compute capability"):
            get_architecture_for_compute_capability("invalid")  # type: ignore[arg-type]


class TestAllDeviceTypes:
    """Test all device types can be created."""

    @pytest.mark.parametrize("device_type", list(VALID_DEVICE_TYPES))
    def test_create_config_with_device_type(self, device_type: str) -> None:
        """Config can be created with each device type."""
        config = create_device_config(device_type=device_type)  # type: ignore[arg-type]
        assert config.device_type.value == device_type


class TestAllGPUArchitectures:
    """Test all GPU architectures can be retrieved."""

    @pytest.mark.parametrize("arch", list(VALID_GPU_ARCHITECTURES))
    def test_get_architecture(self, arch: str) -> None:
        """Architecture can be retrieved."""
        result = get_gpu_architecture(arch)
        assert result.value == arch


class TestAllComputeCapabilities:
    """Test all compute capabilities can be retrieved."""

    @pytest.mark.parametrize("cap", list(VALID_COMPUTE_CAPABILITIES))
    def test_get_capability(self, cap: str) -> None:
        """Capability can be retrieved."""
        result = get_compute_capability(cap)
        assert result.value == cap

    @pytest.mark.parametrize("cap", list(VALID_COMPUTE_CAPABILITIES))
    def test_create_gpu_info_with_capability(self, cap: str) -> None:
        """GPU info can be created with each capability."""
        info = create_gpu_info("Test GPU", 24.0, cap)  # type: ignore[arg-type]
        assert info.compute_capability.value == cap
