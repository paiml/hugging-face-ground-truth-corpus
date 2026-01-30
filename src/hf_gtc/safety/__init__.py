"""Safety recipes for HuggingFace models.

This module provides utilities for watermarking LLM-generated text,
content safety guardrails, PII detection, and toxicity classification.

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
    "VALID_CONTENT_CATEGORIES",
    "VALID_GUARDRAIL_TYPES",
    "ActionType",
    "ContentCategory",
    "ContentPolicyConfig",
    # Watermarking
    "DetectionConfig",
    "DetectionMethod",
    "EmbeddingConfig",
    "GuardrailConfig",
    "GuardrailResult",
    "GuardrailType",
    "PIIConfig",
    "ToxicityConfig",
    "WatermarkConfig",
    "WatermarkResult",
    "WatermarkStats",
    "WatermarkStrength",
    "WatermarkType",
    "calculate_safety_score",
    "calculate_z_score",
    "check_content_safety",
    "create_content_policy_config",
    "create_detection_config",
    "create_embedding_config",
    "create_guardrail_config",
    "create_pii_config",
    "create_toxicity_config",
    "create_watermark_config",
    "estimate_detectability",
    "get_action_type",
    "get_content_category",
    "get_detection_method",
    "get_guardrail_type",
    "get_watermark_strength",
    "get_watermark_type",
    "list_action_types",
    "list_content_categories",
    "list_detection_methods",
    "list_guardrail_types",
    "list_watermark_strengths",
    "list_watermark_types",
    "validate_content_policy_config",
    "validate_detection_config",
    "validate_guardrail_config",
    "validate_watermark_config",
]
