"""Tests for inference.context module."""

from __future__ import annotations

import pytest

from hf_gtc.inference.context import (
    VALID_ATTENTION_PATTERNS,
    VALID_EXTENSION_METHODS,
    VALID_ROPE_SCALING_TYPES,
    AttentionPattern,
    ContextConfig,
    ContextStats,
    ExtensionMethod,
    RoPEConfig,
    RoPEScalingType,
    SlidingWindowConfig,
    calculate_attention_complexity,
    calculate_effective_length,
    create_context_config,
    create_context_stats,
    create_rope_config,
    create_sliding_window_config,
    estimate_memory_scaling,
    format_context_stats,
    get_attention_pattern,
    get_extension_method,
    get_recommended_context_config,
    get_rope_scaling_type,
    list_attention_patterns,
    list_extension_methods,
    list_rope_scaling_types,
    validate_context_config,
    validate_context_stats,
    validate_position_ids,
    validate_rope_config,
    validate_sliding_window_config,
)


class TestExtensionMethod:
    """Tests for ExtensionMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in ExtensionMethod:
            assert isinstance(method.value, str)

    def test_rope_scaling_value(self) -> None:
        """ROPE_SCALING has correct value."""
        assert ExtensionMethod.ROPE_SCALING.value == "rope_scaling"

    def test_alibi_value(self) -> None:
        """ALIBI has correct value."""
        assert ExtensionMethod.ALIBI.value == "alibi"

    def test_sliding_window_value(self) -> None:
        """SLIDING_WINDOW has correct value."""
        assert ExtensionMethod.SLIDING_WINDOW.value == "sliding_window"

    def test_landmark_value(self) -> None:
        """LANDMARK has correct value."""
        assert ExtensionMethod.LANDMARK.value == "landmark"

    def test_streaming_value(self) -> None:
        """STREAMING has correct value."""
        assert ExtensionMethod.STREAMING.value == "streaming"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_EXTENSION_METHODS is a frozenset."""
        assert isinstance(VALID_EXTENSION_METHODS, frozenset)

    def test_valid_methods_contains_all(self) -> None:
        """Frozenset contains all enum values."""
        for method in ExtensionMethod:
            assert method.value in VALID_EXTENSION_METHODS


class TestRoPEScalingType:
    """Tests for RoPEScalingType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for scaling_type in RoPEScalingType:
            assert isinstance(scaling_type.value, str)

    def test_linear_value(self) -> None:
        """LINEAR has correct value."""
        assert RoPEScalingType.LINEAR.value == "linear"

    def test_dynamic_value(self) -> None:
        """DYNAMIC has correct value."""
        assert RoPEScalingType.DYNAMIC.value == "dynamic"

    def test_yarn_value(self) -> None:
        """YARN has correct value."""
        assert RoPEScalingType.YARN.value == "yarn"

    def test_ntk_value(self) -> None:
        """NTK has correct value."""
        assert RoPEScalingType.NTK.value == "ntk"

    def test_valid_types_frozenset(self) -> None:
        """VALID_ROPE_SCALING_TYPES is a frozenset."""
        assert isinstance(VALID_ROPE_SCALING_TYPES, frozenset)

    def test_valid_types_contains_all(self) -> None:
        """Frozenset contains all enum values."""
        for scaling_type in RoPEScalingType:
            assert scaling_type.value in VALID_ROPE_SCALING_TYPES


class TestAttentionPattern:
    """Tests for AttentionPattern enum."""

    def test_all_patterns_have_values(self) -> None:
        """All patterns have string values."""
        for pattern in AttentionPattern:
            assert isinstance(pattern.value, str)

    def test_full_value(self) -> None:
        """FULL has correct value."""
        assert AttentionPattern.FULL.value == "full"

    def test_local_value(self) -> None:
        """LOCAL has correct value."""
        assert AttentionPattern.LOCAL.value == "local"

    def test_global_local_value(self) -> None:
        """GLOBAL_LOCAL has correct value."""
        assert AttentionPattern.GLOBAL_LOCAL.value == "global_local"

    def test_dilated_value(self) -> None:
        """DILATED has correct value."""
        assert AttentionPattern.DILATED.value == "dilated"

    def test_valid_patterns_frozenset(self) -> None:
        """VALID_ATTENTION_PATTERNS is a frozenset."""
        assert isinstance(VALID_ATTENTION_PATTERNS, frozenset)

    def test_valid_patterns_contains_all(self) -> None:
        """Frozenset contains all enum values."""
        for pattern in AttentionPattern:
            assert pattern.value in VALID_ATTENTION_PATTERNS


class TestRoPEConfig:
    """Tests for RoPEConfig dataclass."""

    def test_create_config(self) -> None:
        """Create RoPE config."""
        config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=4096,
        )
        assert config.scaling_factor == pytest.approx(2.0)
        assert config.scaling_type == RoPEScalingType.LINEAR
        assert config.original_max_length == 4096

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=4096,
        )
        with pytest.raises(AttributeError):
            config.scaling_factor = 4.0  # type: ignore[misc]


class TestSlidingWindowConfig:
    """Tests for SlidingWindowConfig dataclass."""

    def test_create_config(self) -> None:
        """Create sliding window config."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=128,
        )
        assert config.window_size == 4096
        assert config.sink_tokens == 4
        assert config.overlap == 128

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=128,
        )
        with pytest.raises(AttributeError):
            config.window_size = 8192  # type: ignore[misc]


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_create_config_with_rope(self) -> None:
        """Create context config with RoPE."""
        rope_config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=4096,
        )
        config = ContextConfig(
            extension_method=ExtensionMethod.ROPE_SCALING,
            rope_config=rope_config,
            window_config=None,
            max_length=8192,
        )
        assert config.max_length == 8192
        assert config.extension_method == ExtensionMethod.ROPE_SCALING
        assert config.rope_config is not None

    def test_create_config_with_window(self) -> None:
        """Create context config with sliding window."""
        window_config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=128,
        )
        config = ContextConfig(
            extension_method=ExtensionMethod.SLIDING_WINDOW,
            rope_config=None,
            window_config=window_config,
            max_length=1000000,
        )
        assert config.max_length == 1000000
        assert config.extension_method == ExtensionMethod.SLIDING_WINDOW
        assert config.window_config is not None

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ContextConfig(
            extension_method=ExtensionMethod.ALIBI,
            rope_config=None,
            window_config=None,
            max_length=8192,
        )
        with pytest.raises(AttributeError):
            config.max_length = 16384  # type: ignore[misc]


class TestContextStats:
    """Tests for ContextStats dataclass."""

    def test_create_stats(self) -> None:
        """Create context stats."""
        stats = ContextStats(
            effective_length=8192,
            memory_usage_mb=1024.0,
            attention_sparsity=0.75,
        )
        assert stats.effective_length == 8192
        assert stats.memory_usage_mb == pytest.approx(1024.0)
        assert stats.attention_sparsity == pytest.approx(0.75)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = ContextStats(8192, 1024.0, 0.75)
        with pytest.raises(AttributeError):
            stats.effective_length = 16384  # type: ignore[misc]


class TestValidateRoPEConfig:
    """Tests for validate_rope_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=4096,
        )
        validate_rope_config(config)

    def test_zero_scaling_factor_raises(self) -> None:
        """Zero scaling factor raises ValueError."""
        config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=0.0,
            original_max_length=4096,
        )
        with pytest.raises(ValueError, match="scaling_factor must be positive"):
            validate_rope_config(config)

    def test_negative_scaling_factor_raises(self) -> None:
        """Negative scaling factor raises ValueError."""
        config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=-1.0,
            original_max_length=4096,
        )
        with pytest.raises(ValueError, match="scaling_factor must be positive"):
            validate_rope_config(config)

    def test_zero_original_max_length_raises(self) -> None:
        """Zero original max length raises ValueError."""
        config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=0,
        )
        with pytest.raises(ValueError, match="original_max_length must be positive"):
            validate_rope_config(config)

    def test_negative_original_max_length_raises(self) -> None:
        """Negative original max length raises ValueError."""
        config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=-1,
        )
        with pytest.raises(ValueError, match="original_max_length must be positive"):
            validate_rope_config(config)


class TestValidateSlidingWindowConfig:
    """Tests for validate_sliding_window_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=128,
        )
        validate_sliding_window_config(config)

    def test_zero_window_size_raises(self) -> None:
        """Zero window size raises ValueError."""
        config = SlidingWindowConfig(
            window_size=0,
            sink_tokens=4,
            overlap=128,
        )
        with pytest.raises(ValueError, match="window_size must be positive"):
            validate_sliding_window_config(config)

    def test_negative_window_size_raises(self) -> None:
        """Negative window size raises ValueError."""
        config = SlidingWindowConfig(
            window_size=-1,
            sink_tokens=4,
            overlap=128,
        )
        with pytest.raises(ValueError, match="window_size must be positive"):
            validate_sliding_window_config(config)

    def test_negative_sink_tokens_raises(self) -> None:
        """Negative sink tokens raises ValueError."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=-1,
            overlap=128,
        )
        with pytest.raises(ValueError, match="sink_tokens cannot be negative"):
            validate_sliding_window_config(config)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=-1,
        )
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            validate_sliding_window_config(config)

    def test_overlap_exceeds_window_raises(self) -> None:
        """Overlap >= window size raises ValueError."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=4096,
        )
        with pytest.raises(ValueError, match="overlap must be less than window_size"):
            validate_sliding_window_config(config)

    def test_overlap_greater_than_window_raises(self) -> None:
        """Overlap greater than window size raises ValueError."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=5000,
        )
        with pytest.raises(ValueError, match="overlap must be less than window_size"):
            validate_sliding_window_config(config)

    def test_zero_sink_tokens_valid(self) -> None:
        """Zero sink tokens is valid."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=0,
            overlap=128,
        )
        validate_sliding_window_config(config)

    def test_zero_overlap_valid(self) -> None:
        """Zero overlap is valid."""
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=0,
        )
        validate_sliding_window_config(config)


class TestValidateContextConfig:
    """Tests for validate_context_config function."""

    def test_valid_rope_config(self) -> None:
        """Valid RoPE config passes validation."""
        rope_config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=4096,
        )
        config = ContextConfig(
            extension_method=ExtensionMethod.ROPE_SCALING,
            rope_config=rope_config,
            window_config=None,
            max_length=8192,
        )
        validate_context_config(config)

    def test_valid_sliding_window_config(self) -> None:
        """Valid sliding window config passes validation."""
        window_config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=128,
        )
        config = ContextConfig(
            extension_method=ExtensionMethod.SLIDING_WINDOW,
            rope_config=None,
            window_config=window_config,
            max_length=1000000,
        )
        validate_context_config(config)

    def test_zero_max_length_raises(self) -> None:
        """Zero max length raises ValueError."""
        config = ContextConfig(
            extension_method=ExtensionMethod.ALIBI,
            rope_config=None,
            window_config=None,
            max_length=0,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_context_config(config)

    def test_negative_max_length_raises(self) -> None:
        """Negative max length raises ValueError."""
        config = ContextConfig(
            extension_method=ExtensionMethod.ALIBI,
            rope_config=None,
            window_config=None,
            max_length=-1,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_context_config(config)

    def test_rope_scaling_without_rope_config_raises(self) -> None:
        """RoPE scaling without rope_config raises ValueError."""
        config = ContextConfig(
            extension_method=ExtensionMethod.ROPE_SCALING,
            rope_config=None,
            window_config=None,
            max_length=8192,
        )
        with pytest.raises(ValueError, match="rope_config is required"):
            validate_context_config(config)

    def test_sliding_window_without_window_config_raises(self) -> None:
        """Sliding window without window_config raises ValueError."""
        config = ContextConfig(
            extension_method=ExtensionMethod.SLIDING_WINDOW,
            rope_config=None,
            window_config=None,
            max_length=8192,
        )
        with pytest.raises(ValueError, match="window_config is required"):
            validate_context_config(config)

    def test_alibi_without_configs_valid(self) -> None:
        """ALiBi method without extra configs is valid."""
        config = ContextConfig(
            extension_method=ExtensionMethod.ALIBI,
            rope_config=None,
            window_config=None,
            max_length=8192,
        )
        validate_context_config(config)


class TestValidateContextStats:
    """Tests for validate_context_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats pass validation."""
        stats = ContextStats(
            effective_length=8192,
            memory_usage_mb=1024.0,
            attention_sparsity=0.75,
        )
        validate_context_stats(stats)

    def test_negative_effective_length_raises(self) -> None:
        """Negative effective length raises ValueError."""
        stats = ContextStats(
            effective_length=-1,
            memory_usage_mb=1024.0,
            attention_sparsity=0.75,
        )
        with pytest.raises(ValueError, match="effective_length cannot be negative"):
            validate_context_stats(stats)

    def test_negative_memory_usage_raises(self) -> None:
        """Negative memory usage raises ValueError."""
        stats = ContextStats(
            effective_length=8192,
            memory_usage_mb=-1.0,
            attention_sparsity=0.75,
        )
        with pytest.raises(ValueError, match="memory_usage_mb cannot be negative"):
            validate_context_stats(stats)

    def test_sparsity_below_zero_raises(self) -> None:
        """Sparsity below 0 raises ValueError."""
        stats = ContextStats(
            effective_length=8192,
            memory_usage_mb=1024.0,
            attention_sparsity=-0.1,
        )
        with pytest.raises(ValueError, match="attention_sparsity must be between"):
            validate_context_stats(stats)

    def test_sparsity_above_one_raises(self) -> None:
        """Sparsity above 1 raises ValueError."""
        stats = ContextStats(
            effective_length=8192,
            memory_usage_mb=1024.0,
            attention_sparsity=1.1,
        )
        with pytest.raises(ValueError, match="attention_sparsity must be between"):
            validate_context_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        stats = ContextStats(
            effective_length=0,
            memory_usage_mb=0.0,
            attention_sparsity=0.0,
        )
        validate_context_stats(stats)

    def test_max_sparsity_valid(self) -> None:
        """Maximum sparsity (1.0) is valid."""
        stats = ContextStats(
            effective_length=8192,
            memory_usage_mb=1024.0,
            attention_sparsity=1.0,
        )
        validate_context_stats(stats)


class TestCreateRoPEConfig:
    """Tests for create_rope_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_rope_config()
        assert config.scaling_type == RoPEScalingType.LINEAR
        assert config.scaling_factor == pytest.approx(2.0)
        assert config.original_max_length == 4096

    def test_custom_scaling_type(self) -> None:
        """Create config with custom scaling type."""
        config = create_rope_config(scaling_type="yarn")
        assert config.scaling_type == RoPEScalingType.YARN

    def test_custom_scaling_factor(self) -> None:
        """Create config with custom scaling factor."""
        config = create_rope_config(scaling_factor=4.0)
        assert config.scaling_factor == pytest.approx(4.0)

    def test_custom_original_max_length(self) -> None:
        """Create config with custom original max length."""
        config = create_rope_config(original_max_length=2048)
        assert config.original_max_length == 2048

    def test_invalid_scaling_type_raises(self) -> None:
        """Invalid scaling type raises ValueError."""
        with pytest.raises(ValueError, match="scaling_type must be one of"):
            create_rope_config(scaling_type="invalid")  # type: ignore[arg-type]

    def test_zero_scaling_factor_raises(self) -> None:
        """Zero scaling factor raises ValueError."""
        with pytest.raises(ValueError, match="scaling_factor must be positive"):
            create_rope_config(scaling_factor=0.0)


class TestCreateSlidingWindowConfig:
    """Tests for create_sliding_window_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_sliding_window_config()
        assert config.window_size == 4096
        assert config.sink_tokens == 4
        assert config.overlap == 128

    def test_custom_window_size(self) -> None:
        """Create config with custom window size."""
        config = create_sliding_window_config(window_size=8192)
        assert config.window_size == 8192

    def test_custom_sink_tokens(self) -> None:
        """Create config with custom sink tokens."""
        config = create_sliding_window_config(sink_tokens=8)
        assert config.sink_tokens == 8

    def test_custom_overlap(self) -> None:
        """Create config with custom overlap."""
        config = create_sliding_window_config(overlap=256)
        assert config.overlap == 256

    def test_zero_window_size_raises(self) -> None:
        """Zero window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            create_sliding_window_config(window_size=0)

    def test_overlap_exceeds_window_raises(self) -> None:
        """Overlap >= window size raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than window_size"):
            create_sliding_window_config(window_size=100, overlap=100)


class TestCreateContextConfig:
    """Tests for create_context_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_context_config()
        assert config.extension_method == ExtensionMethod.ROPE_SCALING
        assert config.max_length == 8192
        assert config.rope_config is not None

    def test_custom_extension_method(self) -> None:
        """Create config with custom extension method."""
        config = create_context_config(extension_method="alibi")
        assert config.extension_method == ExtensionMethod.ALIBI

    def test_custom_max_length(self) -> None:
        """Create config with custom max length."""
        config = create_context_config(max_length=16384)
        assert config.max_length == 16384

    def test_with_rope_config(self) -> None:
        """Create config with provided RoPE config."""
        rope_config = create_rope_config(scaling_factor=4.0)
        config = create_context_config(rope_config=rope_config)
        assert config.rope_config is not None
        assert config.rope_config.scaling_factor == pytest.approx(4.0)

    def test_with_window_config(self) -> None:
        """Create config with provided window config."""
        window_config = create_sliding_window_config(window_size=8192)
        config = create_context_config(
            extension_method="sliding_window", window_config=window_config
        )
        assert config.window_config is not None
        assert config.window_config.window_size == 8192

    def test_auto_creates_rope_config(self) -> None:
        """Auto-creates RoPE config for rope_scaling method."""
        config = create_context_config(extension_method="rope_scaling")
        assert config.rope_config is not None

    def test_auto_creates_window_config(self) -> None:
        """Auto-creates window config for sliding_window method."""
        config = create_context_config(extension_method="sliding_window")
        assert config.window_config is not None

    def test_invalid_extension_method_raises(self) -> None:
        """Invalid extension method raises ValueError."""
        with pytest.raises(ValueError, match="extension_method must be one of"):
            create_context_config(extension_method="invalid")  # type: ignore[arg-type]

    def test_zero_max_length_raises(self) -> None:
        """Zero max length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_context_config(max_length=0)


class TestCreateContextStats:
    """Tests for create_context_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_context_stats()
        assert stats.effective_length == 0
        assert stats.memory_usage_mb == pytest.approx(0.0)
        assert stats.attention_sparsity == pytest.approx(0.0)

    def test_custom_effective_length(self) -> None:
        """Create stats with custom effective length."""
        stats = create_context_stats(effective_length=8192)
        assert stats.effective_length == 8192

    def test_custom_memory_usage(self) -> None:
        """Create stats with custom memory usage."""
        stats = create_context_stats(memory_usage_mb=1024.0)
        assert stats.memory_usage_mb == pytest.approx(1024.0)

    def test_custom_sparsity(self) -> None:
        """Create stats with custom sparsity."""
        stats = create_context_stats(attention_sparsity=0.75)
        assert stats.attention_sparsity == pytest.approx(0.75)

    def test_invalid_sparsity_raises(self) -> None:
        """Invalid sparsity raises ValueError."""
        with pytest.raises(ValueError, match="attention_sparsity must be between"):
            create_context_stats(attention_sparsity=1.5)


class TestListExtensionMethods:
    """Tests for list_extension_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        methods = list_extension_methods()
        assert methods == sorted(methods)

    def test_contains_rope_scaling(self) -> None:
        """Contains rope_scaling."""
        methods = list_extension_methods()
        assert "rope_scaling" in methods

    def test_contains_sliding_window(self) -> None:
        """Contains sliding_window."""
        methods = list_extension_methods()
        assert "sliding_window" in methods

    def test_contains_all_methods(self) -> None:
        """Contains all methods."""
        methods = list_extension_methods()
        assert len(methods) == len(VALID_EXTENSION_METHODS)


class TestListRoPEScalingTypes:
    """Tests for list_rope_scaling_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_rope_scaling_types()
        assert types == sorted(types)

    def test_contains_linear(self) -> None:
        """Contains linear."""
        types = list_rope_scaling_types()
        assert "linear" in types

    def test_contains_yarn(self) -> None:
        """Contains yarn."""
        types = list_rope_scaling_types()
        assert "yarn" in types

    def test_contains_all_types(self) -> None:
        """Contains all types."""
        types = list_rope_scaling_types()
        assert len(types) == len(VALID_ROPE_SCALING_TYPES)


class TestListAttentionPatterns:
    """Tests for list_attention_patterns function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        patterns = list_attention_patterns()
        assert patterns == sorted(patterns)

    def test_contains_full(self) -> None:
        """Contains full."""
        patterns = list_attention_patterns()
        assert "full" in patterns

    def test_contains_local(self) -> None:
        """Contains local."""
        patterns = list_attention_patterns()
        assert "local" in patterns

    def test_contains_all_patterns(self) -> None:
        """Contains all patterns."""
        patterns = list_attention_patterns()
        assert len(patterns) == len(VALID_ATTENTION_PATTERNS)


class TestGetExtensionMethod:
    """Tests for get_extension_method function."""

    def test_get_rope_scaling(self) -> None:
        """Get rope_scaling method."""
        method = get_extension_method("rope_scaling")
        assert method == ExtensionMethod.ROPE_SCALING

    def test_get_alibi(self) -> None:
        """Get alibi method."""
        method = get_extension_method("alibi")
        assert method == ExtensionMethod.ALIBI

    def test_get_sliding_window(self) -> None:
        """Get sliding_window method."""
        method = get_extension_method("sliding_window")
        assert method == ExtensionMethod.SLIDING_WINDOW

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown extension method"):
            get_extension_method("invalid")


class TestGetRoPEScalingType:
    """Tests for get_rope_scaling_type function."""

    def test_get_linear(self) -> None:
        """Get linear type."""
        scaling_type = get_rope_scaling_type("linear")
        assert scaling_type == RoPEScalingType.LINEAR

    def test_get_yarn(self) -> None:
        """Get yarn type."""
        scaling_type = get_rope_scaling_type("yarn")
        assert scaling_type == RoPEScalingType.YARN

    def test_get_dynamic(self) -> None:
        """Get dynamic type."""
        scaling_type = get_rope_scaling_type("dynamic")
        assert scaling_type == RoPEScalingType.DYNAMIC

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown RoPE scaling type"):
            get_rope_scaling_type("invalid")


class TestGetAttentionPattern:
    """Tests for get_attention_pattern function."""

    def test_get_full(self) -> None:
        """Get full pattern."""
        pattern = get_attention_pattern("full")
        assert pattern == AttentionPattern.FULL

    def test_get_local(self) -> None:
        """Get local pattern."""
        pattern = get_attention_pattern("local")
        assert pattern == AttentionPattern.LOCAL

    def test_get_global_local(self) -> None:
        """Get global_local pattern."""
        pattern = get_attention_pattern("global_local")
        assert pattern == AttentionPattern.GLOBAL_LOCAL

    def test_invalid_pattern_raises(self) -> None:
        """Invalid pattern raises ValueError."""
        with pytest.raises(ValueError, match="Unknown attention pattern"):
            get_attention_pattern("invalid")


class TestCalculateEffectiveLength:
    """Tests for calculate_effective_length function."""

    def test_rope_scaling_effective_length(self) -> None:
        """RoPE scaling returns max_length."""
        rope_config = RoPEConfig(
            scaling_type=RoPEScalingType.LINEAR,
            scaling_factor=2.0,
            original_max_length=4096,
        )
        config = ContextConfig(
            extension_method=ExtensionMethod.ROPE_SCALING,
            rope_config=rope_config,
            window_config=None,
            max_length=8192,
        )
        assert calculate_effective_length(config) == 8192

    def test_sliding_window_effective_length(self) -> None:
        """Sliding window returns max_length."""
        window_config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            overlap=128,
        )
        config = ContextConfig(
            extension_method=ExtensionMethod.SLIDING_WINDOW,
            rope_config=None,
            window_config=window_config,
            max_length=1000000,
        )
        assert calculate_effective_length(config) == 1000000

    def test_alibi_effective_length(self) -> None:
        """ALiBi returns max_length."""
        config = ContextConfig(
            extension_method=ExtensionMethod.ALIBI,
            rope_config=None,
            window_config=None,
            max_length=16384,
        )
        assert calculate_effective_length(config) == 16384


class TestEstimateMemoryScaling:
    """Tests for estimate_memory_scaling function."""

    def test_full_attention_quadratic(self) -> None:
        """Full attention scales quadratically."""
        scale = estimate_memory_scaling(4096, 8192, attention_pattern="full")
        assert scale == pytest.approx(4.0)  # (8192/4096)^2 = 4

    def test_double_length_quadruple_memory(self) -> None:
        """Doubling length quadruples memory for full attention."""
        scale = estimate_memory_scaling(1024, 2048, attention_pattern="full")
        assert scale == pytest.approx(4.0)

    def test_local_attention_linear(self) -> None:
        """Local attention scales more efficiently."""
        scale = estimate_memory_scaling(
            4096, 16384, attention_pattern="local", window_size=4096
        )
        # Local: (16384 * 4096) / (4096^2) = 4.0
        assert scale == pytest.approx(4.0)

    def test_global_local_intermediate(self) -> None:
        """Global-local is between full and local."""
        scale_full = estimate_memory_scaling(4096, 16384, attention_pattern="full")
        scale_local = estimate_memory_scaling(
            4096, 16384, attention_pattern="local", window_size=4096
        )
        scale_gl = estimate_memory_scaling(
            4096, 16384, attention_pattern="global_local", window_size=4096
        )
        # Global-local should be between full and local
        assert scale_local <= scale_gl <= scale_full

    def test_zero_original_length_raises(self) -> None:
        """Zero original length raises ValueError."""
        with pytest.raises(ValueError, match="original_length must be positive"):
            estimate_memory_scaling(0, 8192)

    def test_zero_target_length_raises(self) -> None:
        """Zero target length raises ValueError."""
        with pytest.raises(ValueError, match="target_length must be positive"):
            estimate_memory_scaling(4096, 0)

    def test_invalid_attention_pattern_raises(self) -> None:
        """Invalid attention pattern raises ValueError."""
        with pytest.raises(ValueError, match="attention_pattern must be one of"):
            estimate_memory_scaling(4096, 8192, attention_pattern="invalid")  # type: ignore[arg-type]

    def test_local_without_window_raises(self) -> None:
        """Local attention without window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size is required"):
            estimate_memory_scaling(4096, 8192, attention_pattern="local")

    def test_zero_window_size_raises(self) -> None:
        """Zero window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            estimate_memory_scaling(
                4096, 8192, attention_pattern="local", window_size=0
            )

    def test_dilated_attention(self) -> None:
        """Dilated attention reduces memory."""
        scale_full = estimate_memory_scaling(4096, 16384, attention_pattern="full")
        scale_dilated = estimate_memory_scaling(
            4096, 16384, attention_pattern="dilated", window_size=4096
        )
        assert scale_dilated < scale_full


class TestCalculateAttentionComplexity:
    """Tests for calculate_attention_complexity function."""

    def test_full_attention_quadratic(self) -> None:
        """Full attention is O(n^2)."""
        flops = calculate_attention_complexity(4096, attention_pattern="full")
        assert flops > 0
        # With 32 heads: 2 * 32 * 4096^2 = 1073741824
        expected = 2 * 32 * 4096 * 4096
        assert flops == expected

    def test_longer_sequence_more_flops(self) -> None:
        """Longer sequences require more FLOPs."""
        flops_short = calculate_attention_complexity(2048, attention_pattern="full")
        flops_long = calculate_attention_complexity(4096, attention_pattern="full")
        assert flops_long > flops_short

    def test_local_attention_linear(self) -> None:
        """Local attention is O(n * w)."""
        flops_full = calculate_attention_complexity(8192, attention_pattern="full")
        flops_local = calculate_attention_complexity(
            8192, attention_pattern="local", window_size=4096
        )
        assert flops_local < flops_full

    def test_zero_seq_length_raises(self) -> None:
        """Zero sequence length raises ValueError."""
        with pytest.raises(ValueError, match="seq_length must be positive"):
            calculate_attention_complexity(0)

    def test_zero_num_heads_raises(self) -> None:
        """Zero num heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            calculate_attention_complexity(4096, num_heads=0)

    def test_invalid_attention_pattern_raises(self) -> None:
        """Invalid attention pattern raises ValueError."""
        with pytest.raises(ValueError, match="attention_pattern must be one of"):
            calculate_attention_complexity(4096, attention_pattern="invalid")  # type: ignore[arg-type]

    def test_local_without_window_raises(self) -> None:
        """Local attention without window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size is required"):
            calculate_attention_complexity(4096, attention_pattern="local")

    def test_zero_window_size_raises(self) -> None:
        """Zero window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            calculate_attention_complexity(
                4096, attention_pattern="local", window_size=0
            )

    def test_custom_num_heads(self) -> None:
        """Custom number of heads affects FLOPs."""
        flops_32 = calculate_attention_complexity(4096, num_heads=32)
        flops_64 = calculate_attention_complexity(4096, num_heads=64)
        assert flops_64 == flops_32 * 2

    def test_global_local_flops(self) -> None:
        """Global-local FLOPs are between full and local."""
        flops_full = calculate_attention_complexity(8192, attention_pattern="full")
        flops_local = calculate_attention_complexity(
            8192, attention_pattern="local", window_size=2048
        )
        flops_gl = calculate_attention_complexity(
            8192, attention_pattern="global_local", window_size=2048
        )
        assert flops_local < flops_gl < flops_full

    def test_dilated_flops(self) -> None:
        """Dilated attention reduces FLOPs."""
        flops_full = calculate_attention_complexity(8192, attention_pattern="full")
        flops_dilated = calculate_attention_complexity(
            8192, attention_pattern="dilated", window_size=2048
        )
        assert flops_dilated < flops_full


class TestValidatePositionIds:
    """Tests for validate_position_ids function."""

    def test_valid_position_ids(self) -> None:
        """Valid position IDs pass validation."""
        assert validate_position_ids([0, 1, 2, 3], max_length=4096) is True

    def test_empty_list_valid(self) -> None:
        """Empty list is valid."""
        assert validate_position_ids([], max_length=4096) is True

    def test_duplicates_allowed_by_default(self) -> None:
        """Duplicates are allowed by default."""
        assert validate_position_ids([0, 1, 1, 2], max_length=4096) is True

    def test_duplicates_not_allowed(self) -> None:
        """Duplicates raise error when not allowed."""
        with pytest.raises(ValueError, match="Duplicate position IDs found"):
            validate_position_ids([0, 1, 1, 2], max_length=4096, allow_duplicates=False)

    def test_negative_position_raises(self) -> None:
        """Negative position ID raises ValueError."""
        with pytest.raises(ValueError, match="Position ID cannot be negative"):
            validate_position_ids([-1, 0, 1], max_length=4096)

    def test_exceeds_max_length_raises(self) -> None:
        """Position exceeding max length raises ValueError."""
        with pytest.raises(ValueError, match=r"Position ID .* exceeds max_length"):
            validate_position_ids([0, 1, 5000], max_length=4096)

    def test_exactly_at_max_length_raises(self) -> None:
        """Position at max_length raises (must be < max_length)."""
        with pytest.raises(ValueError, match=r"Position ID .* exceeds max_length"):
            validate_position_ids([0, 4096], max_length=4096)

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_position_ids([0], max_length=0)

    def test_negative_max_length_raises(self) -> None:
        """Negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_position_ids([0], max_length=-1)


class TestFormatContextStats:
    """Tests for format_context_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = ContextStats(
            effective_length=8192,
            memory_usage_mb=1024.0,
            attention_sparsity=0.75,
        )
        formatted = format_context_stats(stats)
        assert "Effective Length: 8192" in formatted
        assert "Memory Usage: 1024.00 MB" in formatted
        assert "Attention Sparsity: 75.0%" in formatted

    def test_zero_values(self) -> None:
        """Zero values format correctly."""
        stats = ContextStats(0, 0.0, 0.0)
        formatted = format_context_stats(stats)
        assert "Effective Length: 0" in formatted
        assert "Memory Usage: 0.00 MB" in formatted
        assert "Attention Sparsity: 0.0%" in formatted

    def test_multiline_output(self) -> None:
        """Output contains multiple lines."""
        stats = ContextStats(8192, 1024.0, 0.75)
        formatted = format_context_stats(stats)
        lines = formatted.split("\n")
        assert len(lines) == 3

    def test_full_sparsity(self) -> None:
        """Full sparsity formats correctly."""
        stats = ContextStats(8192, 1024.0, 1.0)
        formatted = format_context_stats(stats)
        assert "Attention Sparsity: 100.0%" in formatted


class TestGetRecommendedContextConfig:
    """Tests for get_recommended_context_config function."""

    def test_small_extension_linear_scaling(self) -> None:
        """Small extension (<=2x) uses linear scaling."""
        config = get_recommended_context_config(target_length=8192)
        assert config.max_length == 8192
        assert config.extension_method == ExtensionMethod.ROPE_SCALING
        assert config.rope_config is not None
        assert config.rope_config.scaling_type == RoPEScalingType.LINEAR

    def test_medium_extension_dynamic_scaling(self) -> None:
        """Medium extension (2x-8x) uses dynamic scaling."""
        config = get_recommended_context_config(
            target_length=16384, original_max_length=4096
        )
        assert config.rope_config is not None
        assert config.rope_config.scaling_type == RoPEScalingType.DYNAMIC

    def test_large_extension_yarn_scaling(self) -> None:
        """Large extension (>8x) uses YaRN scaling."""
        config = get_recommended_context_config(
            target_length=65536, original_max_length=4096
        )
        assert config.rope_config is not None
        assert config.rope_config.scaling_type == RoPEScalingType.YARN

    def test_memory_constrained_sliding_window(self) -> None:
        """Memory constraint triggers sliding window."""
        config = get_recommended_context_config(
            target_length=1000000,
            memory_constraint_gb=8.0,
        )
        assert config.extension_method == ExtensionMethod.SLIDING_WINDOW
        assert config.window_config is not None

    def test_zero_target_length_raises(self) -> None:
        """Zero target length raises ValueError."""
        with pytest.raises(ValueError, match="target_length must be positive"):
            get_recommended_context_config(target_length=0)

    def test_zero_original_max_length_raises(self) -> None:
        """Zero original max length raises ValueError."""
        with pytest.raises(ValueError, match="original_max_length must be positive"):
            get_recommended_context_config(target_length=8192, original_max_length=0)

    def test_zero_memory_constraint_raises(self) -> None:
        """Zero memory constraint raises ValueError."""
        with pytest.raises(ValueError, match="memory_constraint_gb must be positive"):
            get_recommended_context_config(target_length=8192, memory_constraint_gb=0.0)

    def test_negative_memory_constraint_raises(self) -> None:
        """Negative memory constraint raises ValueError."""
        with pytest.raises(ValueError, match="memory_constraint_gb must be positive"):
            get_recommended_context_config(
                target_length=8192, memory_constraint_gb=-1.0
            )

    def test_scaling_factor_calculated(self) -> None:
        """Scaling factor is correctly calculated."""
        config = get_recommended_context_config(
            target_length=8192, original_max_length=4096
        )
        assert config.rope_config is not None
        assert config.rope_config.scaling_factor == pytest.approx(2.0)

    def test_memory_constraint_with_sufficient_memory(self) -> None:
        """Memory constraint with sufficient memory uses RoPE scaling."""
        # Small target length that fits in large memory
        config = get_recommended_context_config(
            target_length=4096,
            memory_constraint_gb=100.0,  # Large enough for full attention
        )
        # Should use RoPE scaling, not sliding window
        assert config.extension_method == ExtensionMethod.ROPE_SCALING
        assert config.rope_config is not None


class TestAllExtensionMethods:
    """Test all extension methods can be created."""

    @pytest.mark.parametrize("method", list(VALID_EXTENSION_METHODS))
    def test_create_config_with_method(self, method: str) -> None:
        """Config can be created with each method."""
        config = create_context_config(extension_method=method)  # type: ignore[arg-type]
        assert config.extension_method.value == method


class TestAllRoPEScalingTypes:
    """Test all RoPE scaling types can be created."""

    @pytest.mark.parametrize("scaling_type", list(VALID_ROPE_SCALING_TYPES))
    def test_create_config_with_scaling_type(self, scaling_type: str) -> None:
        """Config can be created with each scaling type."""
        config = create_rope_config(scaling_type=scaling_type)  # type: ignore[arg-type]
        assert config.scaling_type.value == scaling_type


class TestAllAttentionPatterns:
    """Test all attention patterns."""

    @pytest.mark.parametrize("pattern", list(VALID_ATTENTION_PATTERNS))
    def test_get_attention_pattern(self, pattern: str) -> None:
        """Pattern can be retrieved by name."""
        result = get_attention_pattern(pattern)
        assert result.value == pattern

    @pytest.mark.parametrize("pattern", ["local", "global_local", "dilated"])
    def test_memory_scaling_with_window(self, pattern: str) -> None:
        """Memory scaling works for patterns requiring window."""
        scale = estimate_memory_scaling(
            4096,
            8192,
            attention_pattern=pattern,
            window_size=2048,  # type: ignore[arg-type]
        )
        assert scale > 0

    @pytest.mark.parametrize("pattern", ["local", "global_local", "dilated"])
    def test_attention_complexity_with_window(self, pattern: str) -> None:
        """Attention complexity works for patterns requiring window."""
        flops = calculate_attention_complexity(
            4096,
            attention_pattern=pattern,
            window_size=2048,  # type: ignore[arg-type]
        )
        assert flops > 0
