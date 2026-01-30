"""Safety recipes for HuggingFace models.

This module provides utilities for watermarking LLM-generated text,
content safety guardrails, PII detection, toxicity classification,
differential privacy, and data anonymization.

Examples:
    >>> from hf_gtc.safety import WatermarkType, WatermarkConfig
    >>> wtype = WatermarkType.SOFT
    >>> wtype.value
    'soft'
    >>> config = WatermarkConfig(gamma=0.25, delta=2.0)
    >>> config.gamma
    0.25

    >>> from hf_gtc.safety import GuardrailType, ActionType
    >>> GuardrailType.PII_FILTER.value
    'pii_filter'
    >>> ActionType.REDACT.value
    'redact'

    >>> from hf_gtc.safety import PrivacyMechanism, AnonymizationType
    >>> PrivacyMechanism.LAPLACE.value
    'laplace'
    >>> AnonymizationType.K_ANONYMITY.value
    'k_anonymity'
"""

from __future__ import annotations

from hf_gtc.safety.guardrails import (
    VALID_ACTION_TYPES,
    VALID_CONTENT_CATEGORIES,
    VALID_GUARDRAIL_TYPES,
    ActionType,
    ContentCategory,
    ContentPolicyConfig,
    GuardrailConfig,
    GuardrailResult,
    GuardrailType,
    PIIConfig,
    ToxicityConfig,
    calculate_safety_score,
    check_content_safety,
    create_content_policy_config,
    create_guardrail_config,
    create_pii_config,
    create_toxicity_config,
    get_action_type,
    get_content_category,
    get_guardrail_type,
    list_action_types,
    list_content_categories,
    list_guardrail_types,
    validate_content_policy_config,
    validate_guardrail_config,
)
from hf_gtc.safety.privacy import (
    VALID_ANONYMIZATION_TYPES,
    VALID_PRIVACY_MECHANISMS,
    VALID_SENSITIVITY_TYPES,
    AnonymizationConfig,
    AnonymizationType,
    DPConfig,
    PrivacyConfig,
    PrivacyMechanism,
    PrivacyStats,
    SensitivityType,
    add_differential_privacy_noise,
    calculate_noise_scale,
    check_k_anonymity,
    compute_privacy_budget,
    create_anonymization_config,
    create_dp_config,
    create_privacy_config,
    estimate_utility_loss,
    format_privacy_stats,
    get_anonymization_type,
    get_privacy_mechanism,
    get_recommended_privacy_config,
    get_sensitivity_type,
    list_anonymization_types,
    list_privacy_mechanisms,
    list_sensitivity_types,
    validate_anonymization_config,
    validate_dp_config,
    validate_privacy_config,
)
from hf_gtc.safety.watermarking import (
    DetectionConfig,
    DetectionMethod,
    EmbeddingConfig,
    WatermarkConfig,
    WatermarkResult,
    WatermarkStats,
    WatermarkStrength,
    WatermarkType,
    calculate_z_score,
    create_detection_config,
    create_embedding_config,
    create_watermark_config,
    estimate_detectability,
    get_detection_method,
    get_watermark_strength,
    get_watermark_type,
    list_detection_methods,
    list_watermark_strengths,
    list_watermark_types,
    validate_detection_config,
    validate_watermark_config,
)

__all__: list[str] = [
    # Guardrails
    "VALID_ACTION_TYPES",
    # Privacy
    "VALID_ANONYMIZATION_TYPES",
    "VALID_CONTENT_CATEGORIES",
    "VALID_GUARDRAIL_TYPES",
    "VALID_PRIVACY_MECHANISMS",
    "VALID_SENSITIVITY_TYPES",
    "ActionType",
    "AnonymizationConfig",
    "AnonymizationType",
    "ContentCategory",
    "ContentPolicyConfig",
    "DPConfig",
    # Watermarking
    "DetectionConfig",
    "DetectionMethod",
    "EmbeddingConfig",
    "GuardrailConfig",
    "GuardrailResult",
    "GuardrailType",
    "PIIConfig",
    "PrivacyConfig",
    "PrivacyMechanism",
    "PrivacyStats",
    "SensitivityType",
    "ToxicityConfig",
    "WatermarkConfig",
    "WatermarkResult",
    "WatermarkStats",
    "WatermarkStrength",
    "WatermarkType",
    "add_differential_privacy_noise",
    "calculate_noise_scale",
    "calculate_safety_score",
    "calculate_z_score",
    "check_content_safety",
    "check_k_anonymity",
    "compute_privacy_budget",
    "create_anonymization_config",
    "create_content_policy_config",
    "create_detection_config",
    "create_dp_config",
    "create_embedding_config",
    "create_guardrail_config",
    "create_pii_config",
    "create_privacy_config",
    "create_toxicity_config",
    "create_watermark_config",
    "estimate_detectability",
    "estimate_utility_loss",
    "format_privacy_stats",
    "get_action_type",
    "get_anonymization_type",
    "get_content_category",
    "get_detection_method",
    "get_guardrail_type",
    "get_privacy_mechanism",
    "get_recommended_privacy_config",
    "get_sensitivity_type",
    "get_watermark_strength",
    "get_watermark_type",
    "list_action_types",
    "list_anonymization_types",
    "list_content_categories",
    "list_detection_methods",
    "list_guardrail_types",
    "list_privacy_mechanisms",
    "list_sensitivity_types",
    "list_watermark_strengths",
    "list_watermark_types",
    "validate_anonymization_config",
    "validate_content_policy_config",
    "validate_detection_config",
    "validate_dp_config",
    "validate_guardrail_config",
    "validate_privacy_config",
    "validate_watermark_config",
]
