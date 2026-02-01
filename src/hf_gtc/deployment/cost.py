"""Cost estimation and resource optimization for ML deployments.

This module provides utilities for estimating cloud costs, comparing
providers, and optimizing instance selection for ML workloads.

Examples:
    >>> from hf_gtc.deployment.cost import CloudProvider, InstanceType
    >>> CloudProvider.AWS.value
    'aws'
    >>> InstanceType.GPU_DATACENTER.value
    'gpu_datacenter'

    >>> from hf_gtc.deployment.cost import create_instance_config
    >>> config = create_instance_config(provider="aws", gpu_count=1)
    >>> config.provider
    <CloudProvider.AWS: 'aws'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class CloudProvider(Enum):
    """Cloud providers for ML deployments.

    Attributes:
        AWS: Amazon Web Services.
        GCP: Google Cloud Platform.
        AZURE: Microsoft Azure.
        LAMBDA_LABS: Lambda Labs GPU cloud.
        RUNPOD: RunPod serverless GPU.

    Examples:
        >>> CloudProvider.AWS.value
        'aws'
        >>> CloudProvider.GCP.value
        'gcp'
        >>> CloudProvider.LAMBDA_LABS.value
        'lambda_labs'
    """

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LAMBDA_LABS = "lambda_labs"
    RUNPOD = "runpod"


VALID_CLOUD_PROVIDERS = frozenset(p.value for p in CloudProvider)


class InstanceType(Enum):
    """Instance types for ML deployments.

    Attributes:
        CPU: CPU-only instances.
        GPU_CONSUMER: Consumer-grade GPUs (RTX series).
        GPU_DATACENTER: Datacenter GPUs (A100, H100).
        TPU: Google TPU instances.

    Examples:
        >>> InstanceType.CPU.value
        'cpu'
        >>> InstanceType.GPU_DATACENTER.value
        'gpu_datacenter'
        >>> InstanceType.TPU.value
        'tpu'
    """

    CPU = "cpu"
    GPU_CONSUMER = "gpu_consumer"
    GPU_DATACENTER = "gpu_datacenter"
    TPU = "tpu"


VALID_INSTANCE_TYPES = frozenset(t.value for t in InstanceType)


class PricingModel(Enum):
    """Pricing models for cloud resources.

    Attributes:
        ON_DEMAND: Pay-as-you-go pricing.
        SPOT: Interruptible spot/preemptible instances.
        RESERVED: Reserved capacity with commitment.
        SERVERLESS: Serverless pay-per-use.

    Examples:
        >>> PricingModel.ON_DEMAND.value
        'on_demand'
        >>> PricingModel.SPOT.value
        'spot'
        >>> PricingModel.SERVERLESS.value
        'serverless'
    """

    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    SERVERLESS = "serverless"


VALID_PRICING_MODELS = frozenset(m.value for m in PricingModel)


@dataclass(frozen=True, slots=True)
class InstanceConfig:
    """Configuration for a cloud instance.

    Attributes:
        provider: Cloud provider.
        instance_type: Type of instance.
        gpu_type: GPU model (e.g., "A100", "H100"). None for CPU.
        gpu_count: Number of GPUs. 0 for CPU instances.
        memory_gb: Memory in gigabytes.

    Examples:
        >>> config = InstanceConfig(
        ...     provider=CloudProvider.AWS,
        ...     instance_type=InstanceType.GPU_DATACENTER,
        ...     gpu_type="A100",
        ...     gpu_count=8,
        ...     memory_gb=640,
        ... )
        >>> config.gpu_count
        8
        >>> config.provider
        <CloudProvider.AWS: 'aws'>
    """

    provider: CloudProvider
    instance_type: InstanceType
    gpu_type: str | None
    gpu_count: int
    memory_gb: int


@dataclass(frozen=True, slots=True)
class PricingConfig:
    """Configuration for pricing calculation.

    Attributes:
        pricing_model: Pricing model to use.
        region: Cloud region (e.g., "us-east-1").
        commitment_months: Commitment period for reserved pricing.

    Examples:
        >>> config = PricingConfig(
        ...     pricing_model=PricingModel.ON_DEMAND,
        ...     region="us-east-1",
        ...     commitment_months=0,
        ... )
        >>> config.region
        'us-east-1'
        >>> config.pricing_model
        <PricingModel.ON_DEMAND: 'on_demand'>
    """

    pricing_model: PricingModel
    region: str
    commitment_months: int


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Cost estimate for a deployment.

    Attributes:
        hourly_cost: Cost per hour in USD.
        monthly_cost: Cost per month in USD.
        cost_per_1k_tokens: Cost per 1000 tokens processed.
        total_cost: Total estimated cost for the workload.

    Examples:
        >>> estimate = CostEstimate(
        ...     hourly_cost=3.50,
        ...     monthly_cost=2520.0,
        ...     cost_per_1k_tokens=0.001,
        ...     total_cost=5040.0,
        ... )
        >>> estimate.hourly_cost
        3.5
        >>> estimate.monthly_cost
        2520.0
    """

    hourly_cost: float
    monthly_cost: float
    cost_per_1k_tokens: float
    total_cost: float


@dataclass(frozen=True, slots=True)
class CostStats:
    """Statistics for cost analysis.

    Attributes:
        total_cost: Total cost in USD.
        cost_breakdown: Breakdown by category.
        cost_per_request: Average cost per request.
        utilization: Resource utilization (0.0 to 1.0).

    Examples:
        >>> stats = CostStats(
        ...     total_cost=1000.0,
        ...     cost_breakdown={"compute": 800.0, "storage": 200.0},
        ...     cost_per_request=0.001,
        ...     utilization=0.75,
        ... )
        >>> stats.total_cost
        1000.0
        >>> stats.utilization
        0.75
    """

    total_cost: float
    cost_breakdown: dict[str, float]
    cost_per_request: float
    utilization: float


# List/Get/Validate functions for enums


def list_cloud_providers() -> list[str]:
    """List all available cloud providers.

    Returns:
        Sorted list of cloud provider names.

    Examples:
        >>> providers = list_cloud_providers()
        >>> "aws" in providers
        True
        >>> "gcp" in providers
        True
        >>> providers == sorted(providers)
        True
    """
    return sorted(VALID_CLOUD_PROVIDERS)


def validate_cloud_provider(provider: str) -> bool:
    """Validate if a string is a valid cloud provider.

    Args:
        provider: The provider string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_cloud_provider("aws")
        True
        >>> validate_cloud_provider("gcp")
        True
        >>> validate_cloud_provider("invalid")
        False
        >>> validate_cloud_provider("")
        False
    """
    return provider in VALID_CLOUD_PROVIDERS


def get_cloud_provider(name: str) -> CloudProvider:
    """Get CloudProvider enum from string name.

    Args:
        name: Name of the cloud provider.

    Returns:
        Corresponding CloudProvider enum value.

    Raises:
        ValueError: If name is not a valid cloud provider.

    Examples:
        >>> get_cloud_provider("aws")
        <CloudProvider.AWS: 'aws'>

        >>> get_cloud_provider("lambda_labs")
        <CloudProvider.LAMBDA_LABS: 'lambda_labs'>

        >>> get_cloud_provider("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid cloud provider: invalid
    """
    if not validate_cloud_provider(name):
        msg = f"invalid cloud provider: {name}"
        raise ValueError(msg)

    return CloudProvider(name)


def list_instance_types() -> list[str]:
    """List all available instance types.

    Returns:
        Sorted list of instance type names.

    Examples:
        >>> types = list_instance_types()
        >>> "cpu" in types
        True
        >>> "gpu_datacenter" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_INSTANCE_TYPES)


def validate_instance_type(instance_type: str) -> bool:
    """Validate if a string is a valid instance type.

    Args:
        instance_type: The instance type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_instance_type("cpu")
        True
        >>> validate_instance_type("gpu_datacenter")
        True
        >>> validate_instance_type("invalid")
        False
        >>> validate_instance_type("")
        False
    """
    return instance_type in VALID_INSTANCE_TYPES


def get_instance_type(name: str) -> InstanceType:
    """Get InstanceType enum from string name.

    Args:
        name: Name of the instance type.

    Returns:
        Corresponding InstanceType enum value.

    Raises:
        ValueError: If name is not a valid instance type.

    Examples:
        >>> get_instance_type("cpu")
        <InstanceType.CPU: 'cpu'>

        >>> get_instance_type("gpu_datacenter")
        <InstanceType.GPU_DATACENTER: 'gpu_datacenter'>

        >>> get_instance_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid instance type: invalid
    """
    if not validate_instance_type(name):
        msg = f"invalid instance type: {name}"
        raise ValueError(msg)

    return InstanceType(name)


def list_pricing_models() -> list[str]:
    """List all available pricing models.

    Returns:
        Sorted list of pricing model names.

    Examples:
        >>> models = list_pricing_models()
        >>> "on_demand" in models
        True
        >>> "spot" in models
        True
        >>> models == sorted(models)
        True
    """
    return sorted(VALID_PRICING_MODELS)


def validate_pricing_model(model: str) -> bool:
    """Validate if a string is a valid pricing model.

    Args:
        model: The pricing model string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_pricing_model("on_demand")
        True
        >>> validate_pricing_model("spot")
        True
        >>> validate_pricing_model("invalid")
        False
        >>> validate_pricing_model("")
        False
    """
    return model in VALID_PRICING_MODELS


def get_pricing_model(name: str) -> PricingModel:
    """Get PricingModel enum from string name.

    Args:
        name: Name of the pricing model.

    Returns:
        Corresponding PricingModel enum value.

    Raises:
        ValueError: If name is not a valid pricing model.

    Examples:
        >>> get_pricing_model("on_demand")
        <PricingModel.ON_DEMAND: 'on_demand'>

        >>> get_pricing_model("spot")
        <PricingModel.SPOT: 'spot'>

        >>> get_pricing_model("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid pricing model: invalid
    """
    if not validate_pricing_model(name):
        msg = f"invalid pricing model: {name}"
        raise ValueError(msg)

    return PricingModel(name)


# Validation functions for configs


def validate_instance_config(config: InstanceConfig) -> None:
    """Validate instance configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = InstanceConfig(
        ...     CloudProvider.AWS, InstanceType.GPU_DATACENTER, "A100", 8, 640
        ... )
        >>> validate_instance_config(config)

        >>> bad = InstanceConfig(
        ...     CloudProvider.AWS, InstanceType.GPU_DATACENTER, "A100", -1, 640
        ... )
        >>> validate_instance_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gpu_count must be non-negative, got -1

        >>> bad = InstanceConfig(
        ...     CloudProvider.AWS, InstanceType.GPU_DATACENTER, "A100", 1, 0
        ... )
        >>> validate_instance_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory_gb must be positive, got 0
    """
    if config.gpu_count < 0:
        msg = f"gpu_count must be non-negative, got {config.gpu_count}"
        raise ValueError(msg)

    if config.memory_gb <= 0:
        msg = f"memory_gb must be positive, got {config.memory_gb}"
        raise ValueError(msg)

    _validate_gpu_instance_requirements(config)


def _validate_gpu_instance_requirements(config: InstanceConfig) -> None:
    """Validate GPU-specific requirements for GPU instances."""
    gpu_types = (InstanceType.GPU_CONSUMER, InstanceType.GPU_DATACENTER)
    if config.instance_type not in gpu_types:
        return
    if config.gpu_count == 0:
        msg = "gpu_count must be positive for GPU instances"
        raise ValueError(msg)
    if not config.gpu_type:
        msg = "gpu_type is required for GPU instances"
        raise ValueError(msg)


def validate_pricing_config(config: PricingConfig) -> None:
    """Validate pricing configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = PricingConfig(PricingModel.ON_DEMAND, "us-east-1", 0)
        >>> validate_pricing_config(config)

        >>> bad = PricingConfig(PricingModel.ON_DEMAND, "", 0)
        >>> validate_pricing_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: region cannot be empty

        >>> bad = PricingConfig(PricingModel.RESERVED, "us-east-1", 0)
        >>> validate_pricing_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: commitment_months must be positive for reserved pricing
    """
    if not config.region:
        msg = "region cannot be empty"
        raise ValueError(msg)

    if config.commitment_months < 0:
        msg = f"commitment_months must be non-negative, got {config.commitment_months}"
        raise ValueError(msg)

    if config.pricing_model == PricingModel.RESERVED and config.commitment_months <= 0:
        msg = "commitment_months must be positive for reserved pricing"
        raise ValueError(msg)


def validate_cost_estimate(estimate: CostEstimate) -> None:
    """Validate cost estimate.

    Args:
        estimate: Estimate to validate.

    Raises:
        ValueError: If estimate is invalid.

    Examples:
        >>> estimate = CostEstimate(3.5, 2520.0, 0.001, 5040.0)
        >>> validate_cost_estimate(estimate)

        >>> bad = CostEstimate(-1.0, 2520.0, 0.001, 5040.0)
        >>> validate_cost_estimate(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hourly_cost must be non-negative, got -1.0
    """
    if estimate.hourly_cost < 0:
        msg = f"hourly_cost must be non-negative, got {estimate.hourly_cost}"
        raise ValueError(msg)

    if estimate.monthly_cost < 0:
        msg = f"monthly_cost must be non-negative, got {estimate.monthly_cost}"
        raise ValueError(msg)

    if estimate.cost_per_1k_tokens < 0:
        val = estimate.cost_per_1k_tokens
        msg = f"cost_per_1k_tokens must be non-negative, got {val}"
        raise ValueError(msg)

    if estimate.total_cost < 0:
        msg = f"total_cost must be non-negative, got {estimate.total_cost}"
        raise ValueError(msg)


def validate_cost_stats(stats: CostStats) -> None:
    """Validate cost statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = CostStats(1000.0, {"compute": 800.0}, 0.001, 0.75)
        >>> validate_cost_stats(stats)

        >>> bad = CostStats(-1.0, {}, 0.001, 0.75)
        >>> validate_cost_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_cost must be non-negative, got -1.0
    """
    if stats.total_cost < 0:
        msg = f"total_cost must be non-negative, got {stats.total_cost}"
        raise ValueError(msg)

    if stats.cost_per_request < 0:
        msg = f"cost_per_request must be non-negative, got {stats.cost_per_request}"
        raise ValueError(msg)

    if not 0 <= stats.utilization <= 1:
        msg = f"utilization must be between 0 and 1, got {stats.utilization}"
        raise ValueError(msg)


# Factory functions


def create_instance_config(
    provider: str | CloudProvider = CloudProvider.AWS,
    instance_type: str | InstanceType = InstanceType.GPU_DATACENTER,
    gpu_type: str | None = "A100",
    gpu_count: int = 1,
    memory_gb: int = 80,
) -> InstanceConfig:
    """Create an instance configuration.

    Args:
        provider: Cloud provider. Defaults to AWS.
        instance_type: Type of instance. Defaults to GPU_DATACENTER.
        gpu_type: GPU model. Defaults to "A100".
        gpu_count: Number of GPUs. Defaults to 1.
        memory_gb: Memory in gigabytes. Defaults to 80.

    Returns:
        Validated InstanceConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_instance_config()
        >>> config.provider
        <CloudProvider.AWS: 'aws'>
        >>> config.gpu_count
        1

        >>> config = create_instance_config(provider="gcp", gpu_count=4)
        >>> config.provider
        <CloudProvider.GCP: 'gcp'>
        >>> config.gpu_count
        4

        >>> create_instance_config(gpu_count=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gpu_count must be non-negative, got -1
    """
    if isinstance(provider, str):
        provider = get_cloud_provider(provider)
    if isinstance(instance_type, str):
        instance_type = get_instance_type(instance_type)

    config = InstanceConfig(
        provider=provider,
        instance_type=instance_type,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        memory_gb=memory_gb,
    )
    validate_instance_config(config)
    return config


def create_pricing_config(
    pricing_model: str | PricingModel = PricingModel.ON_DEMAND,
    region: str = "us-east-1",
    commitment_months: int = 0,
) -> PricingConfig:
    """Create a pricing configuration.

    Args:
        pricing_model: Pricing model to use. Defaults to ON_DEMAND.
        region: Cloud region. Defaults to "us-east-1".
        commitment_months: Commitment period for reserved pricing.

    Returns:
        Validated PricingConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_pricing_config()
        >>> config.pricing_model
        <PricingModel.ON_DEMAND: 'on_demand'>
        >>> config.region
        'us-east-1'

        >>> config = create_pricing_config(pricing_model="spot", region="eu-west-1")
        >>> config.pricing_model
        <PricingModel.SPOT: 'spot'>

        >>> create_pricing_config(region="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: region cannot be empty
    """
    if isinstance(pricing_model, str):
        pricing_model = get_pricing_model(pricing_model)

    config = PricingConfig(
        pricing_model=pricing_model,
        region=region,
        commitment_months=commitment_months,
    )
    validate_pricing_config(config)
    return config


def create_cost_estimate(
    hourly_cost: float,
    monthly_cost: float | None = None,
    cost_per_1k_tokens: float = 0.0,
    total_cost: float | None = None,
    hours: int = 720,
) -> CostEstimate:
    """Create a cost estimate.

    Args:
        hourly_cost: Cost per hour in USD.
        monthly_cost: Cost per month. Calculated if None.
        cost_per_1k_tokens: Cost per 1000 tokens.
        total_cost: Total cost. Uses monthly_cost if None.
        hours: Hours per month for calculation. Defaults to 720.

    Returns:
        Validated CostEstimate instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> estimate = create_cost_estimate(hourly_cost=3.5)
        >>> estimate.hourly_cost
        3.5
        >>> estimate.monthly_cost
        2520.0

        >>> estimate = create_cost_estimate(hourly_cost=1.0, monthly_cost=700.0)
        >>> estimate.monthly_cost
        700.0

        >>> create_cost_estimate(hourly_cost=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hourly_cost must be non-negative, got -1.0
    """
    if monthly_cost is None:
        monthly_cost = hourly_cost * hours
    if total_cost is None:
        total_cost = monthly_cost

    estimate = CostEstimate(
        hourly_cost=hourly_cost,
        monthly_cost=monthly_cost,
        cost_per_1k_tokens=cost_per_1k_tokens,
        total_cost=total_cost,
    )
    validate_cost_estimate(estimate)
    return estimate


# Pricing data (simplified reference data)

_GPU_HOURLY_COSTS: dict[str, dict[str, float]] = {
    "A100": {
        "aws": 3.50,
        "gcp": 3.20,
        "azure": 3.40,
        "lambda_labs": 1.10,
        "runpod": 1.44,
    },
    "H100": {
        "aws": 6.00,
        "gcp": 5.50,
        "azure": 5.80,
        "lambda_labs": 2.49,
        "runpod": 3.29,
    },
    "A10G": {
        "aws": 1.50,
        "gcp": 1.30,
        "azure": 1.40,
        "lambda_labs": 0.60,
        "runpod": 0.44,
    },
    "T4": {
        "aws": 0.53,
        "gcp": 0.35,
        "azure": 0.45,
        "lambda_labs": 0.20,
        "runpod": 0.16,
    },
    "RTX4090": {
        "aws": 0.0,
        "gcp": 0.0,
        "azure": 0.0,
        "lambda_labs": 0.74,
        "runpod": 0.44,
    },
}

_PRICING_MODEL_DISCOUNTS: dict[str, float] = {
    "on_demand": 1.0,
    "spot": 0.3,  # 70% discount
    "reserved": 0.6,  # 40% discount for 1-year
    "serverless": 1.2,  # 20% premium for flexibility
}


def _get_base_hourly_cost(gpu_type: str | None, provider: CloudProvider) -> float:
    """Get base hourly cost for a GPU type and provider."""
    if gpu_type is None:
        # CPU instance base cost
        return 0.10

    gpu_costs = _GPU_HOURLY_COSTS.get(gpu_type)
    if gpu_costs is None:
        # Unknown GPU, use A100 as baseline
        gpu_costs = _GPU_HOURLY_COSTS["A100"]

    return gpu_costs.get(provider.value, 3.0)


# Core functions


def estimate_inference_cost(
    instance_config: InstanceConfig,
    pricing_config: PricingConfig,
    tokens_per_second: float = 100.0,
    hours_per_month: int = 720,
) -> CostEstimate:
    """Estimate inference cost for a deployment.

    Args:
        instance_config: Instance configuration.
        pricing_config: Pricing configuration.
        tokens_per_second: Expected throughput in tokens/second.
        hours_per_month: Hours of operation per month.

    Returns:
        Cost estimate for the deployment.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> instance = create_instance_config(
        ...     provider="aws", gpu_type="A100", gpu_count=1
        ... )
        >>> pricing = create_pricing_config(pricing_model="on_demand")
        >>> estimate = estimate_inference_cost(instance, pricing)
        >>> estimate.hourly_cost > 0
        True
        >>> estimate.monthly_cost > estimate.hourly_cost
        True

        >>> estimate_inference_cost(instance, pricing, tokens_per_second=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens_per_second must be positive, got 0
    """
    if tokens_per_second <= 0:
        msg = f"tokens_per_second must be positive, got {tokens_per_second}"
        raise ValueError(msg)

    if hours_per_month <= 0:
        msg = f"hours_per_month must be positive, got {hours_per_month}"
        raise ValueError(msg)

    base_cost = _get_base_hourly_cost(
        instance_config.gpu_type, instance_config.provider
    )
    gpu_multiplier = max(1, instance_config.gpu_count)
    pricing_factor = _PRICING_MODEL_DISCOUNTS.get(
        pricing_config.pricing_model.value, 1.0
    )

    hourly_cost = base_cost * gpu_multiplier * pricing_factor
    monthly_cost = hourly_cost * hours_per_month

    # Calculate cost per 1k tokens
    tokens_per_hour = tokens_per_second * 3600
    cost_per_1k_tokens = (hourly_cost / tokens_per_hour) * 1000

    return CostEstimate(
        hourly_cost=hourly_cost,
        monthly_cost=monthly_cost,
        cost_per_1k_tokens=cost_per_1k_tokens,
        total_cost=monthly_cost,
    )


def estimate_training_cost(
    instance_config: InstanceConfig,
    pricing_config: PricingConfig,
    training_hours: float = 24.0,
    num_epochs: int = 1,
) -> CostEstimate:
    """Estimate training cost for a model.

    Args:
        instance_config: Instance configuration.
        pricing_config: Pricing configuration.
        training_hours: Hours per epoch.
        num_epochs: Number of training epochs.

    Returns:
        Cost estimate for training.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> instance = create_instance_config(
        ...     provider="aws", gpu_type="A100", gpu_count=8
        ... )
        >>> pricing = create_pricing_config(pricing_model="spot")
        >>> estimate = estimate_training_cost(instance, pricing, training_hours=48.0)
        >>> estimate.total_cost > 0
        True

        >>> estimate_training_cost(instance, pricing, training_hours=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: training_hours must be positive, got 0
    """
    if training_hours <= 0:
        msg = f"training_hours must be positive, got {training_hours}"
        raise ValueError(msg)

    if num_epochs <= 0:
        msg = f"num_epochs must be positive, got {num_epochs}"
        raise ValueError(msg)

    base_cost = _get_base_hourly_cost(
        instance_config.gpu_type, instance_config.provider
    )
    gpu_multiplier = max(1, instance_config.gpu_count)
    pricing_factor = _PRICING_MODEL_DISCOUNTS.get(
        pricing_config.pricing_model.value, 1.0
    )

    hourly_cost = base_cost * gpu_multiplier * pricing_factor
    total_hours = training_hours * num_epochs
    total_cost = hourly_cost * total_hours
    monthly_cost = hourly_cost * 720  # Reference monthly cost

    return CostEstimate(
        hourly_cost=hourly_cost,
        monthly_cost=monthly_cost,
        cost_per_1k_tokens=0.0,  # Not applicable for training
        total_cost=total_cost,
    )


def compare_providers(
    gpu_type: str = "A100",
    gpu_count: int = 1,
    pricing_model: str | PricingModel = PricingModel.ON_DEMAND,
) -> dict[str, CostEstimate]:
    """Compare costs across cloud providers.

    Args:
        gpu_type: GPU model to compare.
        gpu_count: Number of GPUs.
        pricing_model: Pricing model to use.

    Returns:
        Dictionary mapping provider names to cost estimates.

    Examples:
        >>> comparison = compare_providers(gpu_type="A100", gpu_count=1)
        >>> "aws" in comparison
        True
        >>> "lambda_labs" in comparison
        True
        >>> all(e.hourly_cost >= 0 for e in comparison.values())
        True
    """
    if isinstance(pricing_model, str):
        pricing_model = get_pricing_model(pricing_model)

    results: dict[str, CostEstimate] = {}

    for provider in CloudProvider:
        instance = InstanceConfig(
            provider=provider,
            instance_type=InstanceType.GPU_DATACENTER,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            memory_gb=80 * gpu_count,
        )
        pricing = PricingConfig(
            pricing_model=pricing_model,
            region="us-east-1",
            commitment_months=0,
        )
        estimate = estimate_inference_cost(instance, pricing)
        results[provider.value] = estimate

    return results


def optimize_instance_selection(
    model_size_gb: float,
    target_throughput: float,
    max_hourly_budget: float,
    preferred_providers: Sequence[str] | None = None,
) -> list[tuple[InstanceConfig, CostEstimate]]:
    """Find optimal instance configurations within budget.

    Args:
        model_size_gb: Model size in gigabytes.
        target_throughput: Target tokens per second.
        max_hourly_budget: Maximum hourly cost in USD.
        preferred_providers: Preferred cloud providers (optional).

    Returns:
        List of (config, estimate) tuples, sorted by cost.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> options = optimize_instance_selection(
        ...     model_size_gb=14.0,
        ...     target_throughput=100.0,
        ...     max_hourly_budget=5.0,
        ... )
        >>> len(options) >= 1
        True
        >>> all(e.hourly_cost <= 5.0 for _, e in options)
        True

        >>> optimize_instance_selection(0, 100.0, 5.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size_gb must be positive, got 0
    """
    if model_size_gb <= 0:
        msg = f"model_size_gb must be positive, got {model_size_gb}"
        raise ValueError(msg)

    if target_throughput <= 0:
        msg = f"target_throughput must be positive, got {target_throughput}"
        raise ValueError(msg)

    if max_hourly_budget <= 0:
        msg = f"max_hourly_budget must be positive, got {max_hourly_budget}"
        raise ValueError(msg)

    providers = (
        [get_cloud_provider(p) for p in preferred_providers]
        if preferred_providers
        else list(CloudProvider)
    )

    # Determine minimum GPU count based on model size
    # Assume ~80GB per A100
    min_gpus_datacenter = max(1, int(model_size_gb / 80) + 1)

    results: list[tuple[InstanceConfig, CostEstimate]] = []

    for provider in providers:
        # Try datacenter GPUs
        for gpu_type in ["A100", "H100", "A10G", "T4"]:
            for gpu_count in range(min_gpus_datacenter, min_gpus_datacenter + 4):
                try:
                    instance = create_instance_config(
                        provider=provider,
                        instance_type=InstanceType.GPU_DATACENTER,
                        gpu_type=gpu_type,
                        gpu_count=gpu_count,
                        memory_gb=80 * gpu_count,
                    )
                    pricing = create_pricing_config()
                    estimate = estimate_inference_cost(
                        instance, pricing, tokens_per_second=target_throughput
                    )

                    if estimate.hourly_cost <= max_hourly_budget:
                        results.append((instance, estimate))
                except ValueError:
                    continue

    # Sort by hourly cost
    results.sort(key=lambda x: x[1].hourly_cost)
    return results


def calculate_cost_per_token(
    hourly_cost: float,
    tokens_per_second: float,
) -> float:
    """Calculate cost per token processed.

    Args:
        hourly_cost: Hourly cost in USD.
        tokens_per_second: Processing throughput.

    Returns:
        Cost per single token in USD.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> cost = calculate_cost_per_token(3.5, 1000.0)
        >>> round(cost, 10)
        9.722e-07

        >>> calculate_cost_per_token(0, 1000.0)
        0.0

        >>> calculate_cost_per_token(3.5, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens_per_second must be positive, got 0
    """
    if hourly_cost < 0:
        msg = f"hourly_cost must be non-negative, got {hourly_cost}"
        raise ValueError(msg)

    if tokens_per_second <= 0:
        msg = f"tokens_per_second must be positive, got {tokens_per_second}"
        raise ValueError(msg)

    if hourly_cost == 0:
        return 0.0

    tokens_per_hour = tokens_per_second * 3600
    return hourly_cost / tokens_per_hour


def format_cost_stats(stats: CostStats) -> str:
    """Format cost statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = CostStats(
        ...     total_cost=1000.0,
        ...     cost_breakdown={"compute": 800.0, "storage": 200.0},
        ...     cost_per_request=0.001,
        ...     utilization=0.75,
        ... )
        >>> formatted = format_cost_stats(stats)
        >>> "Total Cost: $1,000.00" in formatted
        True
        >>> "Utilization: 75.0%" in formatted
        True
    """
    lines = [
        "Cost Statistics:",
        f"  Total Cost: ${stats.total_cost:,.2f}",
        f"  Cost per Request: ${stats.cost_per_request:.6f}",
        f"  Utilization: {stats.utilization * 100:.1f}%",
        "  Breakdown:",
    ]

    for category, cost in sorted(stats.cost_breakdown.items()):
        lines.append(f"    {category}: ${cost:,.2f}")

    return "\n".join(lines)


def get_recommended_cost_config(
    model_size_gb: float,
    use_case: str = "inference",
    budget_tier: str = "standard",
) -> tuple[InstanceConfig, PricingConfig]:
    """Get recommended configuration for a use case.

    Args:
        model_size_gb: Model size in gigabytes.
        use_case: Either "inference" or "training".
        budget_tier: "budget", "standard", or "premium".

    Returns:
        Tuple of (InstanceConfig, PricingConfig).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> inst, pricing = get_recommended_cost_config(7.0, "inference", "standard")
        >>> inst.gpu_count >= 1
        True
        >>> pricing.pricing_model
        <PricingModel.ON_DEMAND: 'on_demand'>

        >>> instance, pricing = get_recommended_cost_config(70.0, "training", "budget")
        >>> instance.gpu_count >= 1
        True
        >>> pricing.pricing_model
        <PricingModel.SPOT: 'spot'>

        >>> get_recommended_cost_config(0, "inference")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size_gb must be positive, got 0
    """
    if model_size_gb <= 0:
        msg = f"model_size_gb must be positive, got {model_size_gb}"
        raise ValueError(msg)

    valid_use_cases = {"inference", "training"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got {use_case}"
        raise ValueError(msg)

    valid_tiers = {"budget", "standard", "premium"}
    if budget_tier not in valid_tiers:
        msg = f"budget_tier must be one of {valid_tiers}, got {budget_tier}"
        raise ValueError(msg)

    # Determine GPU requirements
    if model_size_gb <= 7:
        gpu_type = "T4"
        gpu_count = 1
    elif model_size_gb <= 14:
        gpu_type = "A10G"
        gpu_count = 1
    elif model_size_gb <= 40:
        gpu_type = "A100"
        gpu_count = 1
    elif model_size_gb <= 80:
        gpu_type = "A100"
        gpu_count = 2
    else:
        gpu_type = "H100"
        gpu_count = max(2, int(model_size_gb / 80) + 1)

    # Adjust for training
    if use_case == "training":
        gpu_count = max(gpu_count * 2, 4)

    # Select provider based on tier
    provider_by_tier = {
        "budget": CloudProvider.LAMBDA_LABS,
        "standard": CloudProvider.AWS,
        "premium": CloudProvider.AWS,
    }
    provider = provider_by_tier[budget_tier]

    # Select pricing model based on tier and use case
    if budget_tier == "budget":
        pricing_model = PricingModel.SPOT
    elif budget_tier == "premium":
        if use_case == "inference":
            pricing_model = PricingModel.RESERVED
        else:
            pricing_model = PricingModel.ON_DEMAND
    else:
        pricing_model = PricingModel.ON_DEMAND

    instance = create_instance_config(
        provider=provider,
        instance_type=InstanceType.GPU_DATACENTER,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        memory_gb=80 * gpu_count,
    )

    pricing = create_pricing_config(
        pricing_model=pricing_model,
        region="us-east-1",
        commitment_months=12 if pricing_model == PricingModel.RESERVED else 0,
    )

    return instance, pricing
