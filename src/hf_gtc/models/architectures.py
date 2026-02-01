"""Transformer architecture configurations for HuggingFace models.

This module provides configuration utilities for various transformer architectures
including encoder-only (BERT), decoder-only (GPT, LLaMA), and encoder-decoder (T5)
models, along with utilities for parameter counting and memory estimation.

Examples:
    >>> from hf_gtc.models.architectures import (
    ...     ArchitectureType, create_transformer_config
    ... )
    >>> config = create_transformer_config(num_layers=12, hidden_size=768)
    >>> config.num_layers
    12
    >>> config.hidden_size
    768

    >>> from hf_gtc.models.architectures import ModelFamily, list_model_families
    >>> families = list_model_families()
    >>> "bert" in families
    True
    >>> "llama" in families
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class ArchitectureType(Enum):
    """Transformer architecture types.

    Attributes:
        ENCODER_ONLY: Encoder-only architecture (e.g., BERT).
        DECODER_ONLY: Decoder-only architecture (e.g., GPT, LLaMA).
        ENCODER_DECODER: Encoder-decoder architecture (e.g., T5, BART).
        PREFIX_LM: Prefix language model architecture.

    Examples:
        >>> ArchitectureType.ENCODER_ONLY.value
        'encoder_only'
        >>> ArchitectureType.DECODER_ONLY.value
        'decoder_only'
        >>> ArchitectureType.ENCODER_DECODER.value
        'encoder_decoder'
        >>> ArchitectureType.PREFIX_LM.value
        'prefix_lm'
    """

    ENCODER_ONLY = "encoder_only"
    DECODER_ONLY = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"
    PREFIX_LM = "prefix_lm"


VALID_ARCHITECTURE_TYPES = frozenset(t.value for t in ArchitectureType)


class ModelFamily(Enum):
    """Model family types.

    Attributes:
        BERT: BERT-style models.
        GPT: GPT-style models.
        T5: T5-style models.
        LLAMA: LLaMA-style models.
        MISTRAL: Mistral-style models.
        FALCON: Falcon-style models.

    Examples:
        >>> ModelFamily.BERT.value
        'bert'
        >>> ModelFamily.GPT.value
        'gpt'
        >>> ModelFamily.T5.value
        't5'
        >>> ModelFamily.LLAMA.value
        'llama'
        >>> ModelFamily.MISTRAL.value
        'mistral'
        >>> ModelFamily.FALCON.value
        'falcon'
    """

    BERT = "bert"
    GPT = "gpt"
    T5 = "t5"
    LLAMA = "llama"
    MISTRAL = "mistral"
    FALCON = "falcon"


VALID_MODEL_FAMILIES = frozenset(f.value for f in ModelFamily)


class AttentionPattern(Enum):
    """Attention pattern types.

    Attributes:
        DENSE: Full dense attention (all-to-all).
        SPARSE: Sparse attention pattern.
        LOCAL: Local attention with fixed window.
        SLIDING_WINDOW: Sliding window attention.

    Examples:
        >>> AttentionPattern.DENSE.value
        'dense'
        >>> AttentionPattern.SPARSE.value
        'sparse'
        >>> AttentionPattern.LOCAL.value
        'local'
        >>> AttentionPattern.SLIDING_WINDOW.value
        'sliding_window'
    """

    DENSE = "dense"
    SPARSE = "sparse"
    LOCAL = "local"
    SLIDING_WINDOW = "sliding_window"


VALID_ATTENTION_PATTERNS = frozenset(p.value for p in AttentionPattern)


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Configuration for transformer backbone.

    Attributes:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        vocab_size: Vocabulary size for embeddings.

    Examples:
        >>> config = TransformerConfig(
        ...     num_layers=12,
        ...     hidden_size=768,
        ...     num_heads=12,
        ...     intermediate_size=3072,
        ...     vocab_size=30522,
        ... )
        >>> config.num_layers
        12
        >>> config.hidden_size
        768
        >>> config.num_heads
        12
    """

    num_layers: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Configuration for encoder-only models.

    Attributes:
        transformer_config: Base transformer configuration.
        pooler_type: Type of pooler ("cls", "mean", "max", "none").
        mask_token_id: Token ID for masked language modeling.

    Examples:
        >>> base = TransformerConfig(12, 768, 12, 3072, 30522)
        >>> config = EncoderConfig(
        ...     transformer_config=base,
        ...     pooler_type="cls",
        ...     mask_token_id=103,
        ... )
        >>> config.pooler_type
        'cls'
        >>> config.mask_token_id
        103
    """

    transformer_config: TransformerConfig
    pooler_type: str
    mask_token_id: int


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Configuration for decoder-only models.

    Attributes:
        transformer_config: Base transformer configuration.
        tie_word_embeddings: Whether to tie input/output embeddings.
        use_cache: Whether to use KV caching for inference.

    Examples:
        >>> base = TransformerConfig(32, 4096, 32, 11008, 32000)
        >>> config = DecoderConfig(
        ...     transformer_config=base,
        ...     tie_word_embeddings=False,
        ...     use_cache=True,
        ... )
        >>> config.tie_word_embeddings
        False
        >>> config.use_cache
        True
    """

    transformer_config: TransformerConfig
    tie_word_embeddings: bool
    use_cache: bool


@dataclass(frozen=True, slots=True)
class EncoderDecoderConfig:
    """Configuration for encoder-decoder models.

    Attributes:
        encoder_config: Configuration for the encoder.
        decoder_config: Configuration for the decoder.
        cross_attention: Whether to use cross-attention.

    Examples:
        >>> enc_base = TransformerConfig(12, 768, 12, 3072, 32128)
        >>> dec_base = TransformerConfig(12, 768, 12, 3072, 32128)
        >>> enc = EncoderConfig(enc_base, "none", -1)
        >>> dec = DecoderConfig(dec_base, True, True)
        >>> config = EncoderDecoderConfig(
        ...     encoder_config=enc,
        ...     decoder_config=dec,
        ...     cross_attention=True,
        ... )
        >>> config.cross_attention
        True
    """

    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
    cross_attention: bool


@dataclass(frozen=True, slots=True)
class ArchitectureStats:
    """Statistics for model architecture.

    Attributes:
        total_params: Total parameter count.
        trainable_params: Trainable parameter count.
        memory_footprint_mb: Memory footprint in megabytes.

    Examples:
        >>> stats = ArchitectureStats(
        ...     total_params=110_000_000,
        ...     trainable_params=110_000_000,
        ...     memory_footprint_mb=420.0,
        ... )
        >>> stats.total_params
        110000000
        >>> stats.memory_footprint_mb
        420.0
    """

    total_params: int
    trainable_params: int
    memory_footprint_mb: float


def validate_transformer_config(config: TransformerConfig) -> None:
    """Validate transformer configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_layers is not positive.
        ValueError: If hidden_size is not positive.
        ValueError: If num_heads is not positive.
        ValueError: If hidden_size is not divisible by num_heads.
        ValueError: If intermediate_size is not positive.
        ValueError: If vocab_size is not positive.

    Examples:
        >>> config = TransformerConfig(12, 768, 12, 3072, 30522)
        >>> validate_transformer_config(config)  # No error

        >>> validate_transformer_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = TransformerConfig(0, 768, 12, 3072, 30522)
        >>> validate_transformer_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_layers must be positive
    """
    validate_not_none(config, "config")

    if config.num_layers <= 0:
        msg = f"num_layers must be positive, got {config.num_layers}"
        raise ValueError(msg)

    if config.hidden_size <= 0:
        msg = f"hidden_size must be positive, got {config.hidden_size}"
        raise ValueError(msg)

    if config.num_heads <= 0:
        msg = f"num_heads must be positive, got {config.num_heads}"
        raise ValueError(msg)

    if config.hidden_size % config.num_heads != 0:
        msg = (
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_heads ({config.num_heads})"
        )
        raise ValueError(msg)

    if config.intermediate_size <= 0:
        msg = f"intermediate_size must be positive, got {config.intermediate_size}"
        raise ValueError(msg)

    if config.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {config.vocab_size}"
        raise ValueError(msg)


def validate_encoder_config(config: EncoderConfig) -> None:
    """Validate encoder configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If transformer_config is invalid.
        ValueError: If pooler_type is not valid.

    Examples:
        >>> base = TransformerConfig(12, 768, 12, 3072, 30522)
        >>> config = EncoderConfig(base, "cls", 103)
        >>> validate_encoder_config(config)  # No error

        >>> validate_encoder_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = EncoderConfig(base, "invalid", 103)
        >>> validate_encoder_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pooler_type must be one of...
    """
    validate_not_none(config, "config")

    validate_transformer_config(config.transformer_config)

    valid_poolers = {"cls", "mean", "max", "none"}
    if config.pooler_type not in valid_poolers:
        msg = f"pooler_type must be one of {valid_poolers}, got '{config.pooler_type}'"
        raise ValueError(msg)


def validate_decoder_config(config: DecoderConfig) -> None:
    """Validate decoder configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If transformer_config is invalid.

    Examples:
        >>> base = TransformerConfig(32, 4096, 32, 11008, 32000)
        >>> config = DecoderConfig(base, False, True)
        >>> validate_decoder_config(config)  # No error

        >>> validate_decoder_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_transformer_config(config.transformer_config)


def validate_encoder_decoder_config(config: EncoderDecoderConfig) -> None:
    """Validate encoder-decoder configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If encoder_config is invalid.
        ValueError: If decoder_config is invalid.

    Examples:
        >>> enc_base = TransformerConfig(12, 768, 12, 3072, 32128)
        >>> dec_base = TransformerConfig(12, 768, 12, 3072, 32128)
        >>> enc = EncoderConfig(enc_base, "none", -1)
        >>> dec = DecoderConfig(dec_base, True, True)
        >>> config = EncoderDecoderConfig(enc, dec, True)
        >>> validate_encoder_decoder_config(config)  # No error

        >>> validate_encoder_decoder_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_encoder_config(config.encoder_config)
    validate_decoder_config(config.decoder_config)


def validate_architecture_stats(stats: ArchitectureStats) -> None:
    """Validate architecture statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If total_params is negative.
        ValueError: If trainable_params is negative.
        ValueError: If trainable_params exceeds total_params.
        ValueError: If memory_footprint_mb is negative.

    Examples:
        >>> stats = ArchitectureStats(110_000_000, 110_000_000, 420.0)
        >>> validate_architecture_stats(stats)  # No error

        >>> validate_architecture_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = ArchitectureStats(-1, 0, 0.0)
        >>> validate_architecture_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_params must be non-negative
    """
    validate_not_none(stats, "stats")

    if stats.total_params < 0:
        msg = f"total_params must be non-negative, got {stats.total_params}"
        raise ValueError(msg)

    if stats.trainable_params < 0:
        msg = f"trainable_params must be non-negative, got {stats.trainable_params}"
        raise ValueError(msg)

    if stats.trainable_params > stats.total_params:
        msg = (
            f"trainable_params ({stats.trainable_params}) cannot exceed "
            f"total_params ({stats.total_params})"
        )
        raise ValueError(msg)

    if stats.memory_footprint_mb < 0:
        mem = stats.memory_footprint_mb
        msg = f"memory_footprint_mb must be non-negative, got {mem}"
        raise ValueError(msg)


def create_transformer_config(
    num_layers: int = 12,
    hidden_size: int = 768,
    num_heads: int = 12,
    intermediate_size: int | None = None,
    vocab_size: int = 30522,
) -> TransformerConfig:
    """Create a transformer configuration.

    Args:
        num_layers: Number of transformer layers. Defaults to 12.
        hidden_size: Hidden dimension size. Defaults to 768.
        num_heads: Number of attention heads. Defaults to 12.
        intermediate_size: FFN intermediate dimension. Defaults to 4 * hidden_size.
        vocab_size: Vocabulary size. Defaults to 30522.

    Returns:
        Validated TransformerConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_transformer_config()
        >>> config.num_layers
        12
        >>> config.hidden_size
        768
        >>> config.intermediate_size
        3072

        >>> config = create_transformer_config(
        ...     num_layers=24, hidden_size=1024, num_heads=16
        ... )
        >>> config.num_layers
        24
        >>> config.hidden_size
        1024
        >>> config.intermediate_size
        4096

        >>> create_transformer_config(num_layers=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_layers must be positive
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
    )
    validate_transformer_config(config)
    return config


def create_encoder_config(
    transformer_config: TransformerConfig | None = None,
    pooler_type: str = "cls",
    mask_token_id: int = 103,
) -> EncoderConfig:
    """Create an encoder configuration.

    Args:
        transformer_config: Base transformer config. Defaults to BERT-base config.
        pooler_type: Type of pooler. Defaults to "cls".
        mask_token_id: Token ID for masking. Defaults to 103.

    Returns:
        Validated EncoderConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_encoder_config()
        >>> config.pooler_type
        'cls'
        >>> config.mask_token_id
        103

        >>> base = create_transformer_config(num_layers=6)
        >>> config = create_encoder_config(transformer_config=base, pooler_type="mean")
        >>> config.pooler_type
        'mean'
        >>> config.transformer_config.num_layers
        6

        >>> create_encoder_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     pooler_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: pooler_type must be one of...
    """
    if transformer_config is None:
        transformer_config = create_transformer_config()

    config = EncoderConfig(
        transformer_config=transformer_config,
        pooler_type=pooler_type,
        mask_token_id=mask_token_id,
    )
    validate_encoder_config(config)
    return config


def create_decoder_config(
    transformer_config: TransformerConfig | None = None,
    tie_word_embeddings: bool = False,
    use_cache: bool = True,
) -> DecoderConfig:
    """Create a decoder configuration.

    Args:
        transformer_config: Base transformer config. Defaults to LLaMA-7B config.
        tie_word_embeddings: Whether to tie embeddings. Defaults to False.
        use_cache: Whether to use KV caching. Defaults to True.

    Returns:
        Validated DecoderConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_decoder_config()
        >>> config.tie_word_embeddings
        False
        >>> config.use_cache
        True

        >>> base = create_transformer_config(
        ...     num_layers=32, hidden_size=4096, num_heads=32
        ... )
        >>> config = create_decoder_config(
        ...     transformer_config=base, tie_word_embeddings=True
        ... )
        >>> config.tie_word_embeddings
        True
    """
    if transformer_config is None:
        transformer_config = create_transformer_config(
            num_layers=32,
            hidden_size=4096,
            num_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
        )

    config = DecoderConfig(
        transformer_config=transformer_config,
        tie_word_embeddings=tie_word_embeddings,
        use_cache=use_cache,
    )
    validate_decoder_config(config)
    return config


def create_encoder_decoder_config(
    encoder_config: EncoderConfig | None = None,
    decoder_config: DecoderConfig | None = None,
    cross_attention: bool = True,
) -> EncoderDecoderConfig:
    """Create an encoder-decoder configuration.

    Args:
        encoder_config: Encoder configuration. Defaults to T5 encoder config.
        decoder_config: Decoder configuration. Defaults to T5 decoder config.
        cross_attention: Whether to use cross-attention. Defaults to True.

    Returns:
        Validated EncoderDecoderConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_encoder_decoder_config()
        >>> config.cross_attention
        True

        >>> enc = create_encoder_config(pooler_type="none")
        >>> dec = create_decoder_config(tie_word_embeddings=True)
        >>> config = create_encoder_decoder_config(
        ...     encoder_config=enc,
        ...     decoder_config=dec,
        ...     cross_attention=True,
        ... )
        >>> config.encoder_config.pooler_type
        'none'
    """
    if encoder_config is None:
        enc_base = create_transformer_config(
            num_layers=12,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            vocab_size=32128,
        )
        encoder_config = EncoderConfig(
            transformer_config=enc_base,
            pooler_type="none",
            mask_token_id=-1,
        )

    if decoder_config is None:
        dec_base = create_transformer_config(
            num_layers=12,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            vocab_size=32128,
        )
        decoder_config = DecoderConfig(
            transformer_config=dec_base,
            tie_word_embeddings=True,
            use_cache=True,
        )

    config = EncoderDecoderConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        cross_attention=cross_attention,
    )
    validate_encoder_decoder_config(config)
    return config


def list_architecture_types() -> list[str]:
    """List available architecture types.

    Returns:
        Sorted list of architecture type names.

    Examples:
        >>> types = list_architecture_types()
        >>> "encoder_only" in types
        True
        >>> "decoder_only" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(t.value for t in ArchitectureType)


def list_model_families() -> list[str]:
    """List available model families.

    Returns:
        Sorted list of model family names.

    Examples:
        >>> families = list_model_families()
        >>> "bert" in families
        True
        >>> "llama" in families
        True
        >>> families == sorted(families)
        True
    """
    return sorted(f.value for f in ModelFamily)


def list_attention_patterns() -> list[str]:
    """List available attention patterns.

    Returns:
        Sorted list of attention pattern names.

    Examples:
        >>> patterns = list_attention_patterns()
        >>> "dense" in patterns
        True
        >>> "sliding_window" in patterns
        True
        >>> patterns == sorted(patterns)
        True
    """
    return sorted(p.value for p in AttentionPattern)


def get_architecture_type(name: str) -> ArchitectureType:
    """Get architecture type enum from string.

    Args:
        name: Type name.

    Returns:
        ArchitectureType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_architecture_type("encoder_only")
        <ArchitectureType.ENCODER_ONLY: 'encoder_only'>
        >>> get_architecture_type("decoder_only")
        <ArchitectureType.DECODER_ONLY: 'decoder_only'>

        >>> get_architecture_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: architecture type must be one of...
    """
    for t in ArchitectureType:
        if t.value == name:
            return t
    msg = f"architecture type must be one of {VALID_ARCHITECTURE_TYPES}, got {name}"
    raise ValueError(msg)


def get_model_family(name: str) -> ModelFamily:
    """Get model family enum from string.

    Args:
        name: Family name.

    Returns:
        ModelFamily enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_model_family("bert")
        <ModelFamily.BERT: 'bert'>
        >>> get_model_family("llama")
        <ModelFamily.LLAMA: 'llama'>

        >>> get_model_family("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model family must be one of...
    """
    for f in ModelFamily:
        if f.value == name:
            return f
    msg = f"model family must be one of {VALID_MODEL_FAMILIES}, got {name}"
    raise ValueError(msg)


def get_attention_pattern(name: str) -> AttentionPattern:
    """Get attention pattern enum from string.

    Args:
        name: Pattern name.

    Returns:
        AttentionPattern enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_attention_pattern("dense")
        <AttentionPattern.DENSE: 'dense'>
        >>> get_attention_pattern("sliding_window")
        <AttentionPattern.SLIDING_WINDOW: 'sliding_window'>

        >>> get_attention_pattern("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: attention pattern must be one of...
    """
    for p in AttentionPattern:
        if p.value == name:
            return p
    msg = f"attention pattern must be one of {VALID_ATTENTION_PATTERNS}, got {name}"
    raise ValueError(msg)


def calculate_model_params(config: TransformerConfig) -> int:
    """Calculate total parameter count for a transformer model.

    Computes the approximate number of parameters including embeddings,
    attention layers, and feed-forward networks.

    Args:
        config: Transformer configuration.

    Returns:
        Total parameter count.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_transformer_config(
        ...     num_layers=12, hidden_size=768, num_heads=12,
        ...     intermediate_size=3072, vocab_size=30522
        ... )
        >>> params = calculate_model_params(config)
        >>> params > 100_000_000  # BERT-base ~110M params
        True
        >>> isinstance(params, int)
        True

        >>> calculate_model_params(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_transformer_config(config)

    h = config.hidden_size
    n_layers = config.num_layers
    v = config.vocab_size
    ff = config.intermediate_size

    # Word embeddings: vocab_size * hidden_size
    embedding_params = v * h

    # Per-layer attention: Q, K, V projections + output projection
    # Each is hidden_size * hidden_size = h * h
    # Total per layer: 4 * h * h
    attention_params_per_layer = 4 * h * h

    # Per-layer FFN: up-projection + down-projection
    # up: hidden_size * intermediate_size
    # down: intermediate_size * hidden_size
    ffn_params_per_layer = 2 * h * ff

    # Layer norms: 2 per layer (attention + ffn), each with gamma and beta
    # Each norm: 2 * hidden_size
    norm_params_per_layer = 4 * h

    # Total per layer
    params_per_layer = (
        attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    )

    # All layers
    total_layer_params = n_layers * params_per_layer

    # Final layer norm (some architectures)
    final_norm_params = 2 * h

    return embedding_params + total_layer_params + final_norm_params


def estimate_memory_footprint(
    config: TransformerConfig,
    dtype_bytes: int = 2,
    include_gradients: bool = True,
    include_optimizer: bool = True,
) -> float:
    """Estimate memory footprint in megabytes.

    Args:
        config: Transformer configuration.
        dtype_bytes: Bytes per parameter (2 for fp16, 4 for fp32). Defaults to 2.
        include_gradients: Include gradient memory. Defaults to True.
        include_optimizer: Include optimizer states (Adam). Defaults to True.

    Returns:
        Estimated memory footprint in megabytes.

    Raises:
        ValueError: If config is None.
        ValueError: If dtype_bytes is not valid.

    Examples:
        >>> config = create_transformer_config(
        ...     num_layers=12, hidden_size=768, num_heads=12
        ... )
        >>> mem = estimate_memory_footprint(config)
        >>> mem > 0
        True
        >>> isinstance(mem, float)
        True

        >>> mem_no_grad = estimate_memory_footprint(config, include_gradients=False)
        >>> mem_no_grad < mem
        True

        >>> estimate_memory_footprint(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if dtype_bytes not in (1, 2, 4, 8):
        msg = f"dtype_bytes must be 1, 2, 4, or 8, got {dtype_bytes}"
        raise ValueError(msg)

    num_params = calculate_model_params(config)

    # Model parameters
    model_bytes = num_params * dtype_bytes

    # Gradients (same size as parameters)
    gradient_bytes = num_params * dtype_bytes if include_gradients else 0

    # Optimizer states (Adam: 2 states per parameter at fp32)
    optimizer_bytes = num_params * 4 * 2 if include_optimizer else 0

    total_bytes = model_bytes + gradient_bytes + optimizer_bytes
    return total_bytes / (1024 * 1024)


def compare_architectures(
    configs: list[tuple[str, TransformerConfig]],
) -> dict[str, ArchitectureStats]:
    """Compare multiple architecture configurations.

    Args:
        configs: List of (name, config) tuples to compare.

    Returns:
        Dictionary mapping names to ArchitectureStats.

    Raises:
        ValueError: If configs is empty.
        ValueError: If any config is invalid.

    Examples:
        >>> small = create_transformer_config(
        ...     num_layers=6, hidden_size=512, num_heads=8
        ... )
        >>> large = create_transformer_config(
        ...     num_layers=12, hidden_size=768, num_heads=12
        ... )
        >>> results = compare_architectures([("small", small), ("large", large)])
        >>> "small" in results
        True
        >>> "large" in results
        True
        >>> results["large"].total_params > results["small"].total_params
        True

        >>> compare_architectures([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: configs cannot be empty
    """
    if not configs:
        msg = "configs cannot be empty"
        raise ValueError(msg)

    results = {}
    for name, config in configs:
        if config is None:
            msg = f"config for '{name}' cannot be None"
            raise ValueError(msg)

        total_params = calculate_model_params(config)
        memory_mb = estimate_memory_footprint(
            config, include_gradients=False, include_optimizer=False
        )

        results[name] = ArchitectureStats(
            total_params=total_params,
            trainable_params=total_params,  # All trainable by default
            memory_footprint_mb=memory_mb,
        )

    return results


def get_hidden_states_shape(
    config: TransformerConfig,
    batch_size: int,
    seq_length: int,
) -> tuple[int, int, int]:
    """Get the shape of hidden states tensor.

    Args:
        config: Transformer configuration.
        batch_size: Batch size.
        seq_length: Sequence length.

    Returns:
        Tuple of (batch_size, seq_length, hidden_size).

    Raises:
        ValueError: If config is None.
        ValueError: If batch_size is not positive.
        ValueError: If seq_length is not positive.

    Examples:
        >>> config = create_transformer_config(hidden_size=768)
        >>> shape = get_hidden_states_shape(config, batch_size=4, seq_length=512)
        >>> shape
        (4, 512, 768)

        >>> get_hidden_states_shape(None, 1, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> get_hidden_states_shape(config, 0, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    validate_not_none(config, "config")

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if seq_length <= 0:
        msg = f"seq_length must be positive, got {seq_length}"
        raise ValueError(msg)

    return (batch_size, seq_length, config.hidden_size)


_PARAM_THRESHOLDS = (
    (1e12, 1e12, "T"),
    (1e9, 1e9, "B"),
    (1e6, 1e6, "M"),
    (1e3, 1e3, "K"),
)


def _format_params(count: int) -> str:
    """Format parameter count with appropriate unit."""
    for threshold, divisor, suffix in _PARAM_THRESHOLDS:
        if count >= threshold:
            return f"{count / divisor:.2f}{suffix}"
    return str(count)


def _format_memory(mb: float) -> str:
    """Format memory size with appropriate unit."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def format_architecture_stats(stats: ArchitectureStats) -> str:
    """Format architecture statistics for display.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string with stats breakdown.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = ArchitectureStats(110_000_000, 110_000_000, 420.0)
        >>> formatted = format_architecture_stats(stats)
        >>> "Total Parameters:" in formatted
        True
        >>> "Trainable Parameters:" in formatted
        True
        >>> "Memory Footprint:" in formatted
        True

        >>> format_architecture_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        f"Total Parameters: {_format_params(stats.total_params)}",
        f"Trainable Parameters: {_format_params(stats.trainable_params)}",
        f"Memory Footprint: {_format_memory(stats.memory_footprint_mb)}",
    ]

    return "\n".join(lines)


def get_recommended_architecture_config(
    model_size: str,
    architecture_type: str = "decoder_only",
) -> TransformerConfig:
    """Get recommended architecture configuration for model size.

    Args:
        model_size: Model size string (e.g., "7b", "13b", "70b").
        architecture_type: Architecture type. Defaults to "decoder_only".

    Returns:
        Recommended TransformerConfig.

    Raises:
        ValueError: If model_size is not recognized.
        ValueError: If architecture_type is not valid.

    Examples:
        >>> config = get_recommended_architecture_config("7b")
        >>> config.num_layers
        32
        >>> config.hidden_size
        4096

        >>> config = get_recommended_architecture_config("70b")
        >>> config.num_layers
        80
        >>> config.hidden_size
        8192

        >>> config = get_recommended_architecture_config(  # doctest: +ELLIPSIS
        ...     "base", architecture_type="encoder_only"
        ... )
        >>> config.num_layers
        12
        >>> config.hidden_size
        768

        >>> get_recommended_architecture_config("invalid")
        Traceback (most recent call last):
            ...
        ValueError: unrecognized model size: invalid
    """
    model_size = model_size.lower().strip()

    if architecture_type not in VALID_ARCHITECTURE_TYPES:
        msg = (
            f"architecture_type must be one of {VALID_ARCHITECTURE_TYPES}, "
            f"got '{architecture_type}'"
        )
        raise ValueError(msg)

    # Decoder-only (LLaMA-style) configurations
    decoder_configs = {
        "7b": {
            "num_layers": 32,
            "hidden_size": 4096,
            "num_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        },
        "13b": {
            "num_layers": 40,
            "hidden_size": 5120,
            "num_heads": 40,
            "intermediate_size": 13824,
            "vocab_size": 32000,
        },
        "70b": {
            "num_layers": 80,
            "hidden_size": 8192,
            "num_heads": 64,
            "intermediate_size": 28672,
            "vocab_size": 32000,
        },
    }

    # Encoder-only (BERT-style) configurations
    encoder_configs = {
        "base": {
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 30522,
        },
        "large": {
            "num_layers": 24,
            "hidden_size": 1024,
            "num_heads": 16,
            "intermediate_size": 4096,
            "vocab_size": 30522,
        },
    }

    if architecture_type == "encoder_only":
        configs = encoder_configs
    else:
        configs = decoder_configs

    if model_size not in configs:
        msg = f"unrecognized model size: {model_size}"
        raise ValueError(msg)

    cfg = configs[model_size]
    return create_transformer_config(**cfg)
