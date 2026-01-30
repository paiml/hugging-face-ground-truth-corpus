"""Model architecture recipes for HuggingFace transformers.

This module provides utilities for configuring and analyzing model
architectures including attention mechanisms, activation functions,
normalization layers, positional encodings, neural network layers,
and transformer configurations.

Examples:
    >>> from hf_gtc.models import AttentionType, create_attention_config
    >>> config = create_attention_config(attention_type="flash", num_heads=32)
    >>> config.attention_type
    <AttentionType.FLASH: 'flash'>

    >>> from hf_gtc.models import ActivationType, create_activation_config
    >>> config = create_activation_config(activation_type="gelu")
    >>> config.activation_type
    <ActivationType.GELU: 'gelu'>

    >>> from hf_gtc.models import NormType, create_layer_norm_config
    >>> config = create_layer_norm_config(768)
    >>> config.normalized_shape
    (768,)

    >>> from hf_gtc.models import PositionalType, create_rope_config
    >>> config = create_rope_config(dim=64, max_position=4096)
    >>> config.dim
    64

    >>> from hf_gtc.models import LayerType, create_mlp_config
    >>> config = create_mlp_config(hidden_dim=768, intermediate_dim=3072)
    >>> config.hidden_dim
    768

    >>> from hf_gtc.models import ArchitectureType, create_transformer_config
    >>> config = create_transformer_config(num_layers=12, hidden_size=768)
    >>> config.num_layers
    12
"""

from __future__ import annotations

from hf_gtc.models.activations import (
    VALID_ACTIVATION_TYPES,
    VALID_GELU_APPROXIMATIONS,
    VALID_GLU_VARIANTS,
    ActivationConfig,
    ActivationStats,
    ActivationType,
    GELUApproximation,
    GELUConfig,
    GLUVariant,
    SwiGLUConfig,
    calculate_activation_memory,
    calculate_gradient_magnitude,
    compare_activation_properties,
    create_activation_config,
    create_activation_stats,
    create_gelu_config,
    create_swiglu_config,
    estimate_glu_expansion,
    format_activation_stats,
    get_activation_type,
    get_gelu_approximation,
    get_glu_variant,
    get_recommended_activation_config,
    list_activation_types,
    list_gelu_approximations,
    list_glu_variants,
    validate_activation_config,
    validate_activation_stats,
    validate_gelu_config,
    validate_swiglu_config,
)
from hf_gtc.models.architectures import (
    VALID_ARCHITECTURE_TYPES,
    VALID_ATTENTION_PATTERNS,
    VALID_MODEL_FAMILIES,
    ArchitectureStats,
    ArchitectureType,
    AttentionPattern,
    DecoderConfig,
    EncoderConfig,
    EncoderDecoderConfig,
    ModelFamily,
    TransformerConfig,
    calculate_model_params,
    compare_architectures,
    create_decoder_config,
    create_encoder_config,
    create_encoder_decoder_config,
    create_transformer_config,
    estimate_memory_footprint,
    format_architecture_stats,
    get_architecture_type,
    get_attention_pattern,
    get_hidden_states_shape,
    get_model_family,
    get_recommended_architecture_config,
    list_architecture_types,
    list_attention_patterns,
    list_model_families,
    validate_architecture_stats,
    validate_decoder_config,
    validate_encoder_config,
    validate_encoder_decoder_config,
    validate_transformer_config,
)
from hf_gtc.models.attention import (
    VALID_ATTENTION_IMPLEMENTATIONS,
    VALID_ATTENTION_MASKS,
    VALID_ATTENTION_TYPES,
    AttentionConfig,
    AttentionImplementation,
    AttentionMask,
    AttentionStats,
    AttentionType,
    FlashAttentionConfig,
    GroupedQueryConfig,
    calculate_attention_memory,
    calculate_kv_cache_size,
    create_attention_config,
    create_attention_stats,
    create_flash_attention_config,
    create_grouped_query_config,
    estimate_attention_flops,
    format_attention_stats,
    get_attention_implementation,
    get_attention_mask,
    get_attention_type,
    get_recommended_attention_config,
    list_attention_implementations,
    list_attention_masks,
    list_attention_types,
    select_attention_implementation,
    validate_attention_config,
    validate_flash_attention_config,
    validate_grouped_query_config,
)
from hf_gtc.models.layers import (
    VALID_GATING_TYPES,
    VALID_LAYER_TYPES,
    VALID_PROJECTION_TYPES,
    CrossAttentionConfig,
    GatedMLPConfig,
    GatingType,
    LayerConfig,
    LayerStats,
    LayerType,
    MLPConfig,
    ProjectionType,
    calculate_layer_memory,
    calculate_layer_params,
    compare_layer_configs,
    create_cross_attention_config,
    create_gated_mlp_config,
    create_layer_config,
    create_mlp_config,
    estimate_layer_flops,
    format_layer_stats,
    get_gating_type,
    get_layer_type,
    get_projection_type,
    get_recommended_layer_config,
    list_gating_types,
    list_layer_types,
    list_projection_types,
    validate_cross_attention_config,
    validate_gated_mlp_config,
    validate_layer_config,
    validate_layer_stats,
    validate_mlp_config,
)
from hf_gtc.models.normalization import (
    VALID_EPS_TYPES,
    VALID_NORM_POSITIONS,
    VALID_NORM_TYPES,
    BatchNormConfig,
    EpsType,
    GroupNormConfig,
    LayerNormConfig,
    NormConfig,
    NormPosition,
    NormStats,
    NormType,
    RMSNormConfig,
    calculate_norm_params,
    compare_norm_stability,
    create_batch_norm_config,
    create_group_norm_config,
    create_layer_norm_config,
    create_norm_config,
    create_rms_norm_config,
    estimate_norm_memory,
    format_norm_stats,
    get_eps_type,
    get_eps_value,
    get_norm_position,
    get_norm_type,
    get_recommended_norm_config,
    list_eps_types,
    list_norm_positions,
    list_norm_types,
    select_eps_for_dtype,
    validate_batch_norm_config,
    validate_group_norm_config,
    validate_layer_norm_config,
    validate_norm_config,
    validate_rms_norm_config,
)
from hf_gtc.models.positional import (
    VALID_INTERPOLATION_TYPES,
    VALID_POSITIONAL_TYPES,
    VALID_ROPE_SCALINGS,
    ALiBiConfig,
    InterpolationType,
    PositionalConfig,
    PositionalType,
    RoPEConfig,
    RoPEScaling,
    SinusoidalConfig,
    calculate_alibi_slopes,
    calculate_rope_frequencies,
    calculate_sinusoidal_embeddings,
    create_alibi_config,
    create_positional_config,
    create_rope_config,
    create_sinusoidal_config,
    estimate_position_memory,
    format_positional_stats,
    get_interpolation_type,
    get_positional_type,
    get_recommended_positional_config,
    get_rope_scaling,
    list_interpolation_types,
    list_positional_types,
    list_rope_scalings,
    validate_alibi_config,
    validate_positional_config,
    validate_rope_config,
    validate_sinusoidal_config,
)

__all__: list[str] = [
    # Activation Constants
    "VALID_ACTIVATION_TYPES",
    # Architecture Constants
    "VALID_ARCHITECTURE_TYPES",
    # Attention Constants
    "VALID_ATTENTION_IMPLEMENTATIONS",
    "VALID_ATTENTION_MASKS",
    "VALID_ATTENTION_PATTERNS",
    "VALID_ATTENTION_TYPES",
    # Normalization Constants
    "VALID_EPS_TYPES",
    # Layer Constants
    "VALID_GATING_TYPES",
    "VALID_GELU_APPROXIMATIONS",
    "VALID_GLU_VARIANTS",
    # Positional Constants
    "VALID_INTERPOLATION_TYPES",
    "VALID_LAYER_TYPES",
    # Model Family Constants
    "VALID_MODEL_FAMILIES",
    "VALID_NORM_POSITIONS",
    "VALID_NORM_TYPES",
    "VALID_POSITIONAL_TYPES",
    "VALID_PROJECTION_TYPES",
    "VALID_ROPE_SCALINGS",
    # Positional Dataclasses
    "ALiBiConfig",
    # Activation Dataclasses
    "ActivationConfig",
    "ActivationStats",
    # Activation Enums
    "ActivationType",
    # Architecture Dataclasses
    "ArchitectureStats",
    # Architecture Enums
    "ArchitectureType",
    # Attention Dataclasses
    "AttentionConfig",
    # Attention Enums
    "AttentionImplementation",
    "AttentionMask",
    "AttentionPattern",
    "AttentionStats",
    "AttentionType",
    # Normalization Dataclasses
    "BatchNormConfig",
    # Layer Dataclasses
    "CrossAttentionConfig",
    # Architecture Config Dataclasses
    "DecoderConfig",
    "EncoderConfig",
    "EncoderDecoderConfig",
    # Normalization Enums
    "EpsType",
    "FlashAttentionConfig",
    "GELUApproximation",
    "GELUConfig",
    "GLUVariant",
    "GatedMLPConfig",
    # Layer Enums
    "GatingType",
    "GroupNormConfig",
    "GroupedQueryConfig",
    # Positional Enums
    "InterpolationType",
    "LayerConfig",
    "LayerNormConfig",
    "LayerStats",
    "LayerType",
    "MLPConfig",
    # Model Family Enums
    "ModelFamily",
    "NormConfig",
    "NormPosition",
    "NormStats",
    "NormType",
    "PositionalConfig",
    "PositionalType",
    "ProjectionType",
    "RMSNormConfig",
    "RoPEConfig",
    "RoPEScaling",
    "SinusoidalConfig",
    "SwiGLUConfig",
    # Transformer Config Dataclass
    "TransformerConfig",
    # Activation Calculation functions
    "calculate_activation_memory",
    # Positional Calculation functions
    "calculate_alibi_slopes",
    # Attention Calculation functions
    "calculate_attention_memory",
    "calculate_gradient_magnitude",
    "calculate_kv_cache_size",
    # Layer Calculation functions
    "calculate_layer_memory",
    "calculate_layer_params",
    # Architecture Calculation functions
    "calculate_model_params",
    # Normalization Calculation functions
    "calculate_norm_params",
    "calculate_rope_frequencies",
    "calculate_sinusoidal_embeddings",
    "compare_activation_properties",
    # Architecture Compare functions
    "compare_architectures",
    # Layer Compare functions
    "compare_layer_configs",
    "compare_norm_stability",
    # Activation Factory functions
    "create_activation_config",
    "create_activation_stats",
    # Positional Factory functions
    "create_alibi_config",
    # Attention Factory functions
    "create_attention_config",
    "create_attention_stats",
    # Normalization Factory functions
    "create_batch_norm_config",
    # Layer Factory functions
    "create_cross_attention_config",
    # Architecture Factory functions
    "create_decoder_config",
    "create_encoder_config",
    "create_encoder_decoder_config",
    "create_flash_attention_config",
    "create_gated_mlp_config",
    "create_gelu_config",
    "create_group_norm_config",
    "create_grouped_query_config",
    "create_layer_config",
    "create_layer_norm_config",
    "create_mlp_config",
    "create_norm_config",
    "create_positional_config",
    "create_rms_norm_config",
    "create_rope_config",
    "create_sinusoidal_config",
    "create_swiglu_config",
    # Transformer Config Factory
    "create_transformer_config",
    "estimate_attention_flops",
    "estimate_glu_expansion",
    # Layer Estimation functions
    "estimate_layer_flops",
    # Architecture Memory Estimation
    "estimate_memory_footprint",
    "estimate_norm_memory",
    "estimate_position_memory",
    # Activation Format functions
    "format_activation_stats",
    # Architecture Format functions
    "format_architecture_stats",
    # Attention Format functions
    "format_attention_stats",
    # Layer Format functions
    "format_layer_stats",
    # Normalization Format functions
    "format_norm_stats",
    # Positional Format functions
    "format_positional_stats",
    # Activation Getter functions
    "get_activation_type",
    # Architecture Getter functions
    "get_architecture_type",
    # Attention Getter functions
    "get_attention_implementation",
    "get_attention_mask",
    "get_attention_pattern",
    "get_attention_type",
    # Normalization Getter functions
    "get_eps_type",
    "get_eps_value",
    # Layer Getter functions
    "get_gating_type",
    "get_gelu_approximation",
    "get_glu_variant",
    # Architecture Shape functions
    "get_hidden_states_shape",
    # Positional Getter functions
    "get_interpolation_type",
    "get_layer_type",
    # Model Family Getter functions
    "get_model_family",
    "get_norm_position",
    "get_norm_type",
    "get_positional_type",
    "get_projection_type",
    "get_recommended_activation_config",
    # Architecture Recommended Config functions
    "get_recommended_architecture_config",
    "get_recommended_attention_config",
    # Layer Recommended Config functions
    "get_recommended_layer_config",
    "get_recommended_norm_config",
    "get_recommended_positional_config",
    "get_rope_scaling",
    # Activation List functions
    "list_activation_types",
    # Architecture List functions
    "list_architecture_types",
    # Attention List functions
    "list_attention_implementations",
    "list_attention_masks",
    "list_attention_patterns",
    "list_attention_types",
    # Normalization List functions
    "list_eps_types",
    # Layer List functions
    "list_gating_types",
    "list_gelu_approximations",
    "list_glu_variants",
    # Positional List functions
    "list_interpolation_types",
    "list_layer_types",
    # Model Family List functions
    "list_model_families",
    "list_norm_positions",
    "list_norm_types",
    "list_positional_types",
    "list_projection_types",
    "list_rope_scalings",
    "select_attention_implementation",
    "select_eps_for_dtype",
    # Activation Validation functions
    "validate_activation_config",
    "validate_activation_stats",
    # Positional Validation functions
    "validate_alibi_config",
    # Architecture Validation functions
    "validate_architecture_stats",
    # Attention Validation functions
    "validate_attention_config",
    # Normalization Validation functions
    "validate_batch_norm_config",
    # Layer Validation functions
    "validate_cross_attention_config",
    "validate_decoder_config",
    "validate_encoder_config",
    "validate_encoder_decoder_config",
    "validate_flash_attention_config",
    "validate_gated_mlp_config",
    "validate_gelu_config",
    "validate_group_norm_config",
    "validate_grouped_query_config",
    "validate_layer_config",
    "validate_layer_norm_config",
    "validate_layer_stats",
    "validate_mlp_config",
    "validate_norm_config",
    "validate_positional_config",
    "validate_rms_norm_config",
    "validate_rope_config",
    "validate_sinusoidal_config",
    "validate_swiglu_config",
    "validate_transformer_config",
]
