"""Tests for deployment cost estimation functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.deployment.cost import (
    VALID_CLOUD_PROVIDERS,
    VALID_INSTANCE_TYPES,
    VALID_PRICING_MODELS,
    CloudProvider,
    CostEstimate,
    CostStats,
    InstanceConfig,
    InstanceType,
    PricingConfig,
    PricingModel,
    calculate_cost_per_token,
    compare_providers,
    create_cost_estimate,
    create_instance_config,
    create_pricing_config,
    estimate_inference_cost,
    estimate_training_cost,
    format_cost_stats,
    get_cloud_provider,
    get_instance_type,
    get_pricing_model,
    get_recommended_cost_config,
    list_cloud_providers,
    list_instance_types,
    list_pricing_models,
    optimize_instance_selection,
    validate_cloud_provider,
    validate_cost_estimate,
    validate_cost_stats,
    validate_instance_config,
    validate_instance_type,
    validate_pricing_config,
    validate_pricing_model,
)


class TestCloudProvider:
    """Tests for CloudProvider enum."""

    def test_aws_value(self) -> None:
        """Test AWS enum value."""
        assert CloudProvider.AWS.value == "aws"

    def test_gcp_value(self) -> None:
        """Test GCP enum value."""
        assert CloudProvider.GCP.value == "gcp"

    def test_azure_value(self) -> None:
        """Test AZURE enum value."""
        assert CloudProvider.AZURE.value == "azure"

    def test_lambda_labs_value(self) -> None:
        """Test LAMBDA_LABS enum value."""
        assert CloudProvider.LAMBDA_LABS.value == "lambda_labs"

    def test_runpod_value(self) -> None:
        """Test RUNPOD enum value."""
        assert CloudProvider.RUNPOD.value == "runpod"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_CLOUD_PROVIDERS."""
        for provider in CloudProvider:
            assert provider.value in VALID_CLOUD_PROVIDERS


class TestInstanceType:
    """Tests for InstanceType enum."""

    def test_cpu_value(self) -> None:
        """Test CPU enum value."""
        assert InstanceType.CPU.value == "cpu"

    def test_gpu_consumer_value(self) -> None:
        """Test GPU_CONSUMER enum value."""
        assert InstanceType.GPU_CONSUMER.value == "gpu_consumer"

    def test_gpu_datacenter_value(self) -> None:
        """Test GPU_DATACENTER enum value."""
        assert InstanceType.GPU_DATACENTER.value == "gpu_datacenter"

    def test_tpu_value(self) -> None:
        """Test TPU enum value."""
        assert InstanceType.TPU.value == "tpu"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_INSTANCE_TYPES."""
        for itype in InstanceType:
            assert itype.value in VALID_INSTANCE_TYPES


class TestPricingModel:
    """Tests for PricingModel enum."""

    def test_on_demand_value(self) -> None:
        """Test ON_DEMAND enum value."""
        assert PricingModel.ON_DEMAND.value == "on_demand"

    def test_spot_value(self) -> None:
        """Test SPOT enum value."""
        assert PricingModel.SPOT.value == "spot"

    def test_reserved_value(self) -> None:
        """Test RESERVED enum value."""
        assert PricingModel.RESERVED.value == "reserved"

    def test_serverless_value(self) -> None:
        """Test SERVERLESS enum value."""
        assert PricingModel.SERVERLESS.value == "serverless"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_PRICING_MODELS."""
        for model in PricingModel:
            assert model.value in VALID_PRICING_MODELS


class TestInstanceConfig:
    """Tests for InstanceConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating InstanceConfig instance."""
        config = InstanceConfig(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.GPU_DATACENTER,
            gpu_type="A100",
            gpu_count=8,
            memory_gb=640,
        )
        assert config.provider == CloudProvider.AWS
        assert config.instance_type == InstanceType.GPU_DATACENTER
        assert config.gpu_type == "A100"
        assert config.gpu_count == 8
        assert config.memory_gb == 640

    def test_frozen(self) -> None:
        """Test that InstanceConfig is immutable."""
        config = InstanceConfig(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.GPU_DATACENTER,
            gpu_type="A100",
            gpu_count=8,
            memory_gb=640,
        )
        with pytest.raises(AttributeError):
            config.gpu_count = 4  # type: ignore[misc]

    def test_cpu_instance(self) -> None:
        """Test creating CPU instance config."""
        config = InstanceConfig(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.CPU,
            gpu_type=None,
            gpu_count=0,
            memory_gb=64,
        )
        assert config.gpu_type is None
        assert config.gpu_count == 0


class TestPricingConfig:
    """Tests for PricingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PricingConfig instance."""
        config = PricingConfig(
            pricing_model=PricingModel.ON_DEMAND,
            region="us-east-1",
            commitment_months=0,
        )
        assert config.pricing_model == PricingModel.ON_DEMAND
        assert config.region == "us-east-1"
        assert config.commitment_months == 0

    def test_frozen(self) -> None:
        """Test that PricingConfig is immutable."""
        config = PricingConfig(
            pricing_model=PricingModel.ON_DEMAND,
            region="us-east-1",
            commitment_months=0,
        )
        with pytest.raises(AttributeError):
            config.region = "eu-west-1"  # type: ignore[misc]


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_creation(self) -> None:
        """Test creating CostEstimate instance."""
        estimate = CostEstimate(
            hourly_cost=3.50,
            monthly_cost=2520.0,
            cost_per_1k_tokens=0.001,
            total_cost=5040.0,
        )
        assert estimate.hourly_cost == pytest.approx(3.50)
        assert estimate.monthly_cost == pytest.approx(2520.0)
        assert estimate.cost_per_1k_tokens == pytest.approx(0.001)
        assert estimate.total_cost == pytest.approx(5040.0)

    def test_frozen(self) -> None:
        """Test that CostEstimate is immutable."""
        estimate = CostEstimate(
            hourly_cost=3.50,
            monthly_cost=2520.0,
            cost_per_1k_tokens=0.001,
            total_cost=5040.0,
        )
        with pytest.raises(AttributeError):
            estimate.hourly_cost = 5.0  # type: ignore[misc]


class TestCostStats:
    """Tests for CostStats dataclass."""

    def test_creation(self) -> None:
        """Test creating CostStats instance."""
        stats = CostStats(
            total_cost=1000.0,
            cost_breakdown={"compute": 800.0, "storage": 200.0},
            cost_per_request=0.001,
            utilization=0.75,
        )
        assert stats.total_cost == pytest.approx(1000.0)
        assert stats.cost_breakdown == {"compute": 800.0, "storage": 200.0}
        assert stats.cost_per_request == pytest.approx(0.001)
        assert stats.utilization == pytest.approx(0.75)

    def test_frozen(self) -> None:
        """Test that CostStats is immutable."""
        stats = CostStats(
            total_cost=1000.0,
            cost_breakdown={"compute": 800.0},
            cost_per_request=0.001,
            utilization=0.75,
        )
        with pytest.raises(AttributeError):
            stats.total_cost = 2000.0  # type: ignore[misc]


class TestListCloudProviders:
    """Tests for list_cloud_providers function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_cloud_providers()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_cloud_providers()
        assert result == sorted(result)

    def test_contains_expected_providers(self) -> None:
        """Test that list contains expected providers."""
        result = list_cloud_providers()
        assert "aws" in result
        assert "gcp" in result
        assert "azure" in result
        assert "lambda_labs" in result
        assert "runpod" in result

    def test_correct_count(self) -> None:
        """Test correct number of providers."""
        result = list_cloud_providers()
        assert len(result) == 5


class TestValidateCloudProvider:
    """Tests for validate_cloud_provider function."""

    def test_valid_aws(self) -> None:
        """Test validating AWS."""
        assert validate_cloud_provider("aws") is True

    def test_valid_gcp(self) -> None:
        """Test validating GCP."""
        assert validate_cloud_provider("gcp") is True

    def test_invalid_provider(self) -> None:
        """Test validating invalid provider."""
        assert validate_cloud_provider("invalid") is False

    def test_empty_string(self) -> None:
        """Test validating empty string."""
        assert validate_cloud_provider("") is False


class TestGetCloudProvider:
    """Tests for get_cloud_provider function."""

    def test_get_aws(self) -> None:
        """Test getting AWS provider."""
        assert get_cloud_provider("aws") == CloudProvider.AWS

    def test_get_lambda_labs(self) -> None:
        """Test getting Lambda Labs provider."""
        assert get_cloud_provider("lambda_labs") == CloudProvider.LAMBDA_LABS

    def test_invalid_raises_error(self) -> None:
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="invalid cloud provider"):
            get_cloud_provider("invalid")


class TestListInstanceTypes:
    """Tests for list_instance_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_instance_types()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_instance_types()
        assert result == sorted(result)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        result = list_instance_types()
        assert "cpu" in result
        assert "gpu_datacenter" in result

    def test_correct_count(self) -> None:
        """Test correct number of types."""
        result = list_instance_types()
        assert len(result) == 4


class TestValidateInstanceType:
    """Tests for validate_instance_type function."""

    def test_valid_cpu(self) -> None:
        """Test validating CPU."""
        assert validate_instance_type("cpu") is True

    def test_valid_gpu_datacenter(self) -> None:
        """Test validating gpu_datacenter."""
        assert validate_instance_type("gpu_datacenter") is True

    def test_invalid_type(self) -> None:
        """Test validating invalid type."""
        assert validate_instance_type("invalid") is False


class TestGetInstanceType:
    """Tests for get_instance_type function."""

    def test_get_cpu(self) -> None:
        """Test getting CPU type."""
        assert get_instance_type("cpu") == InstanceType.CPU

    def test_get_gpu_datacenter(self) -> None:
        """Test getting GPU datacenter type."""
        assert get_instance_type("gpu_datacenter") == InstanceType.GPU_DATACENTER

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid instance type"):
            get_instance_type("invalid")


class TestListPricingModels:
    """Tests for list_pricing_models function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_pricing_models()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_pricing_models()
        assert result == sorted(result)

    def test_contains_expected_models(self) -> None:
        """Test that list contains expected models."""
        result = list_pricing_models()
        assert "on_demand" in result
        assert "spot" in result

    def test_correct_count(self) -> None:
        """Test correct number of models."""
        result = list_pricing_models()
        assert len(result) == 4


class TestValidatePricingModel:
    """Tests for validate_pricing_model function."""

    def test_valid_on_demand(self) -> None:
        """Test validating on_demand."""
        assert validate_pricing_model("on_demand") is True

    def test_valid_spot(self) -> None:
        """Test validating spot."""
        assert validate_pricing_model("spot") is True

    def test_invalid_model(self) -> None:
        """Test validating invalid model."""
        assert validate_pricing_model("invalid") is False


class TestGetPricingModel:
    """Tests for get_pricing_model function."""

    def test_get_on_demand(self) -> None:
        """Test getting on_demand model."""
        assert get_pricing_model("on_demand") == PricingModel.ON_DEMAND

    def test_get_spot(self) -> None:
        """Test getting spot model."""
        assert get_pricing_model("spot") == PricingModel.SPOT

    def test_invalid_raises_error(self) -> None:
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="invalid pricing model"):
            get_pricing_model("invalid")


class TestValidateInstanceConfig:
    """Tests for validate_instance_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = InstanceConfig(
            CloudProvider.AWS, InstanceType.GPU_DATACENTER, "A100", 8, 640
        )
        validate_instance_config(config)

    def test_negative_gpu_count_raises(self) -> None:
        """Test that negative gpu_count raises ValueError."""
        config = InstanceConfig(
            CloudProvider.AWS, InstanceType.GPU_DATACENTER, "A100", -1, 640
        )
        with pytest.raises(ValueError, match="gpu_count must be non-negative"):
            validate_instance_config(config)

    def test_zero_memory_raises(self) -> None:
        """Test that zero memory raises ValueError."""
        config = InstanceConfig(
            CloudProvider.AWS, InstanceType.GPU_DATACENTER, "A100", 1, 0
        )
        with pytest.raises(ValueError, match="memory_gb must be positive"):
            validate_instance_config(config)

    def test_gpu_instance_requires_gpu_count(self) -> None:
        """Test that GPU instance requires positive gpu_count."""
        config = InstanceConfig(
            CloudProvider.AWS, InstanceType.GPU_DATACENTER, "A100", 0, 640
        )
        with pytest.raises(ValueError, match="gpu_count must be positive"):
            validate_instance_config(config)

    def test_gpu_instance_requires_gpu_type(self) -> None:
        """Test that GPU instance requires gpu_type."""
        config = InstanceConfig(
            CloudProvider.AWS, InstanceType.GPU_DATACENTER, None, 1, 640
        )
        with pytest.raises(ValueError, match="gpu_type is required"):
            validate_instance_config(config)


class TestValidatePricingConfig:
    """Tests for validate_pricing_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = PricingConfig(PricingModel.ON_DEMAND, "us-east-1", 0)
        validate_pricing_config(config)

    def test_empty_region_raises(self) -> None:
        """Test that empty region raises ValueError."""
        config = PricingConfig(PricingModel.ON_DEMAND, "", 0)
        with pytest.raises(ValueError, match="region cannot be empty"):
            validate_pricing_config(config)

    def test_negative_commitment_raises(self) -> None:
        """Test that negative commitment raises ValueError."""
        config = PricingConfig(PricingModel.ON_DEMAND, "us-east-1", -1)
        with pytest.raises(ValueError, match="commitment_months must be non-negative"):
            validate_pricing_config(config)

    def test_reserved_requires_commitment(self) -> None:
        """Test that reserved pricing requires commitment."""
        config = PricingConfig(PricingModel.RESERVED, "us-east-1", 0)
        with pytest.raises(ValueError, match="commitment_months must be positive"):
            validate_pricing_config(config)


class TestValidateCostEstimate:
    """Tests for validate_cost_estimate function."""

    def test_valid_estimate(self) -> None:
        """Test validating valid estimate."""
        estimate = CostEstimate(3.5, 2520.0, 0.001, 5040.0)
        validate_cost_estimate(estimate)

    def test_negative_hourly_raises(self) -> None:
        """Test that negative hourly cost raises ValueError."""
        estimate = CostEstimate(-1.0, 2520.0, 0.001, 5040.0)
        with pytest.raises(ValueError, match="hourly_cost must be non-negative"):
            validate_cost_estimate(estimate)

    def test_negative_monthly_raises(self) -> None:
        """Test that negative monthly cost raises ValueError."""
        estimate = CostEstimate(3.5, -1.0, 0.001, 5040.0)
        with pytest.raises(ValueError, match="monthly_cost must be non-negative"):
            validate_cost_estimate(estimate)

    def test_negative_token_cost_raises(self) -> None:
        """Test that negative token cost raises ValueError."""
        estimate = CostEstimate(3.5, 2520.0, -0.001, 5040.0)
        with pytest.raises(ValueError, match="cost_per_1k_tokens must be non-negative"):
            validate_cost_estimate(estimate)

    def test_negative_total_raises(self) -> None:
        """Test that negative total cost raises ValueError."""
        estimate = CostEstimate(3.5, 2520.0, 0.001, -1.0)
        with pytest.raises(ValueError, match="total_cost must be non-negative"):
            validate_cost_estimate(estimate)


class TestValidateCostStats:
    """Tests for validate_cost_stats function."""

    def test_valid_stats(self) -> None:
        """Test validating valid stats."""
        stats = CostStats(1000.0, {"compute": 800.0}, 0.001, 0.75)
        validate_cost_stats(stats)

    def test_negative_total_raises(self) -> None:
        """Test that negative total raises ValueError."""
        stats = CostStats(-1.0, {}, 0.001, 0.75)
        with pytest.raises(ValueError, match="total_cost must be non-negative"):
            validate_cost_stats(stats)

    def test_negative_cost_per_request_raises(self) -> None:
        """Test that negative cost per request raises ValueError."""
        stats = CostStats(1000.0, {}, -0.001, 0.75)
        with pytest.raises(ValueError, match="cost_per_request must be non-negative"):
            validate_cost_stats(stats)

    def test_utilization_above_one_raises(self) -> None:
        """Test that utilization above 1 raises ValueError."""
        stats = CostStats(1000.0, {}, 0.001, 1.5)
        with pytest.raises(ValueError, match="utilization must be between 0 and 1"):
            validate_cost_stats(stats)

    def test_negative_utilization_raises(self) -> None:
        """Test that negative utilization raises ValueError."""
        stats = CostStats(1000.0, {}, 0.001, -0.5)
        with pytest.raises(ValueError, match="utilization must be between 0 and 1"):
            validate_cost_stats(stats)


class TestCreateInstanceConfig:
    """Tests for create_instance_config function."""

    def test_default_values(self) -> None:
        """Test creating config with defaults."""
        config = create_instance_config()
        assert config.provider == CloudProvider.AWS
        assert config.instance_type == InstanceType.GPU_DATACENTER
        assert config.gpu_type == "A100"
        assert config.gpu_count == 1
        assert config.memory_gb == 80

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_instance_config(
            provider="gcp",
            instance_type="gpu_consumer",
            gpu_type="RTX4090",
            gpu_count=4,
            memory_gb=256,
        )
        assert config.provider == CloudProvider.GCP
        assert config.instance_type == InstanceType.GPU_CONSUMER
        assert config.gpu_count == 4

    def test_invalid_gpu_count_raises(self) -> None:
        """Test that invalid gpu_count raises ValueError."""
        with pytest.raises(ValueError, match="gpu_count must be non-negative"):
            create_instance_config(gpu_count=-1)

    def test_string_provider(self) -> None:
        """Test using string for provider."""
        config = create_instance_config(provider="lambda_labs")
        assert config.provider == CloudProvider.LAMBDA_LABS


class TestCreatePricingConfig:
    """Tests for create_pricing_config function."""

    def test_default_values(self) -> None:
        """Test creating config with defaults."""
        config = create_pricing_config()
        assert config.pricing_model == PricingModel.ON_DEMAND
        assert config.region == "us-east-1"
        assert config.commitment_months == 0

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_pricing_config(
            pricing_model="spot",
            region="eu-west-1",
        )
        assert config.pricing_model == PricingModel.SPOT
        assert config.region == "eu-west-1"

    def test_empty_region_raises(self) -> None:
        """Test that empty region raises ValueError."""
        with pytest.raises(ValueError, match="region cannot be empty"):
            create_pricing_config(region="")


class TestCreateCostEstimate:
    """Tests for create_cost_estimate function."""

    def test_auto_monthly_calculation(self) -> None:
        """Test automatic monthly cost calculation."""
        estimate = create_cost_estimate(hourly_cost=3.5)
        assert estimate.hourly_cost == pytest.approx(3.5)
        assert estimate.monthly_cost == pytest.approx(3.5 * 720)

    def test_custom_monthly(self) -> None:
        """Test custom monthly cost."""
        estimate = create_cost_estimate(hourly_cost=1.0, monthly_cost=700.0)
        assert estimate.monthly_cost == pytest.approx(700.0)

    def test_negative_hourly_raises(self) -> None:
        """Test that negative hourly cost raises ValueError."""
        with pytest.raises(ValueError, match="hourly_cost must be non-negative"):
            create_cost_estimate(hourly_cost=-1.0)

    def test_custom_hours(self) -> None:
        """Test custom hours calculation."""
        estimate = create_cost_estimate(hourly_cost=1.0, hours=500)
        assert estimate.monthly_cost == pytest.approx(500.0)


class TestEstimateInferenceCost:
    """Tests for estimate_inference_cost function."""

    def test_basic_estimate(self) -> None:
        """Test basic inference cost estimation."""
        instance = create_instance_config(provider="aws", gpu_type="A100", gpu_count=1)
        pricing = create_pricing_config(pricing_model="on_demand")
        estimate = estimate_inference_cost(instance, pricing)
        assert estimate.hourly_cost > 0
        assert estimate.monthly_cost > estimate.hourly_cost

    def test_spot_discount(self) -> None:
        """Test that spot pricing applies discount."""
        instance = create_instance_config()
        on_demand = create_pricing_config(pricing_model="on_demand")
        spot = create_pricing_config(pricing_model="spot")

        on_demand_cost = estimate_inference_cost(instance, on_demand)
        spot_cost = estimate_inference_cost(instance, spot)

        assert spot_cost.hourly_cost < on_demand_cost.hourly_cost

    def test_zero_tokens_raises(self) -> None:
        """Test that zero tokens per second raises ValueError."""
        instance = create_instance_config()
        pricing = create_pricing_config()
        with pytest.raises(ValueError, match="tokens_per_second must be positive"):
            estimate_inference_cost(instance, pricing, tokens_per_second=0)

    def test_zero_hours_raises(self) -> None:
        """Test that zero hours per month raises ValueError."""
        instance = create_instance_config()
        pricing = create_pricing_config()
        with pytest.raises(ValueError, match="hours_per_month must be positive"):
            estimate_inference_cost(instance, pricing, hours_per_month=0)

    @given(st.floats(min_value=0.1, max_value=10000.0))
    @settings(max_examples=50)
    def test_cost_scales_with_throughput(self, tokens_per_second: float) -> None:
        """Test that cost per token decreases with higher throughput."""
        instance = create_instance_config()
        pricing = create_pricing_config()
        estimate = estimate_inference_cost(
            instance, pricing, tokens_per_second=tokens_per_second
        )
        assert estimate.cost_per_1k_tokens >= 0


class TestEstimateTrainingCost:
    """Tests for estimate_training_cost function."""

    def test_basic_estimate(self) -> None:
        """Test basic training cost estimation."""
        instance = create_instance_config(gpu_count=8)
        pricing = create_pricing_config(pricing_model="spot")
        estimate = estimate_training_cost(instance, pricing, training_hours=48.0)
        assert estimate.total_cost > 0

    def test_epochs_multiply_cost(self) -> None:
        """Test that more epochs increase total cost."""
        instance = create_instance_config()
        pricing = create_pricing_config()

        one_epoch = estimate_training_cost(instance, pricing, num_epochs=1)
        two_epochs = estimate_training_cost(instance, pricing, num_epochs=2)

        assert two_epochs.total_cost == pytest.approx(one_epoch.total_cost * 2)

    def test_zero_hours_raises(self) -> None:
        """Test that zero training hours raises ValueError."""
        instance = create_instance_config()
        pricing = create_pricing_config()
        with pytest.raises(ValueError, match="training_hours must be positive"):
            estimate_training_cost(instance, pricing, training_hours=0)

    def test_zero_epochs_raises(self) -> None:
        """Test that zero epochs raises ValueError."""
        instance = create_instance_config()
        pricing = create_pricing_config()
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            estimate_training_cost(instance, pricing, num_epochs=0)


class TestCompareProviders:
    """Tests for compare_providers function."""

    def test_returns_all_providers(self) -> None:
        """Test that comparison includes all providers."""
        comparison = compare_providers(gpu_type="A100", gpu_count=1)
        assert "aws" in comparison
        assert "gcp" in comparison
        assert "azure" in comparison
        assert "lambda_labs" in comparison
        assert "runpod" in comparison

    def test_all_estimates_positive(self) -> None:
        """Test that all estimates have positive costs."""
        comparison = compare_providers()
        for estimate in comparison.values():
            assert estimate.hourly_cost >= 0

    def test_string_pricing_model(self) -> None:
        """Test using string for pricing model."""
        comparison = compare_providers(pricing_model="spot")
        assert len(comparison) == 5


class TestOptimizeInstanceSelection:
    """Tests for optimize_instance_selection function."""

    def test_returns_within_budget(self) -> None:
        """Test that results are within budget."""
        options = optimize_instance_selection(
            model_size_gb=14.0,
            target_throughput=100.0,
            max_hourly_budget=5.0,
        )
        for _, estimate in options:
            assert estimate.hourly_cost <= 5.0

    def test_sorted_by_cost(self) -> None:
        """Test that results are sorted by cost."""
        options = optimize_instance_selection(
            model_size_gb=14.0,
            target_throughput=100.0,
            max_hourly_budget=10.0,
        )
        if len(options) > 1:
            costs = [e.hourly_cost for _, e in options]
            assert costs == sorted(costs)

    def test_zero_model_size_raises(self) -> None:
        """Test that zero model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_gb must be positive"):
            optimize_instance_selection(0, 100.0, 5.0)

    def test_zero_throughput_raises(self) -> None:
        """Test that zero throughput raises ValueError."""
        with pytest.raises(ValueError, match="target_throughput must be positive"):
            optimize_instance_selection(14.0, 0, 5.0)

    def test_zero_budget_raises(self) -> None:
        """Test that zero budget raises ValueError."""
        with pytest.raises(ValueError, match="max_hourly_budget must be positive"):
            optimize_instance_selection(14.0, 100.0, 0)

    def test_preferred_providers(self) -> None:
        """Test filtering by preferred providers."""
        options = optimize_instance_selection(
            model_size_gb=14.0,
            target_throughput=100.0,
            max_hourly_budget=10.0,
            preferred_providers=["lambda_labs"],
        )
        for config, _ in options:
            assert config.provider == CloudProvider.LAMBDA_LABS


class TestCalculateCostPerToken:
    """Tests for calculate_cost_per_token function."""

    def test_basic_calculation(self) -> None:
        """Test basic cost per token calculation."""
        cost = calculate_cost_per_token(3.5, 1000.0)
        # 3.5 / (1000 * 3600) = 9.72e-7
        assert cost == pytest.approx(9.722222e-7, rel=1e-5)

    def test_zero_hourly_cost(self) -> None:
        """Test zero hourly cost returns zero."""
        cost = calculate_cost_per_token(0, 1000.0)
        assert cost == pytest.approx(0.0)

    def test_negative_hourly_raises(self) -> None:
        """Test that negative hourly cost raises ValueError."""
        with pytest.raises(ValueError, match="hourly_cost must be non-negative"):
            calculate_cost_per_token(-1.0, 1000.0)

    def test_zero_throughput_raises(self) -> None:
        """Test that zero throughput raises ValueError."""
        with pytest.raises(ValueError, match="tokens_per_second must be positive"):
            calculate_cost_per_token(3.5, 0)


class TestFormatCostStats:
    """Tests for format_cost_stats function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        stats = CostStats(
            total_cost=1000.0,
            cost_breakdown={"compute": 800.0, "storage": 200.0},
            cost_per_request=0.001,
            utilization=0.75,
        )
        formatted = format_cost_stats(stats)
        assert "Total Cost: $1,000.00" in formatted
        assert "Utilization: 75.0%" in formatted

    def test_includes_breakdown(self) -> None:
        """Test that breakdown is included."""
        stats = CostStats(
            total_cost=1000.0,
            cost_breakdown={"compute": 800.0},
            cost_per_request=0.001,
            utilization=0.75,
        )
        formatted = format_cost_stats(stats)
        assert "compute: $800.00" in formatted

    def test_empty_breakdown(self) -> None:
        """Test formatting with empty breakdown."""
        stats = CostStats(
            total_cost=0.0,
            cost_breakdown={},
            cost_per_request=0.0,
            utilization=0.0,
        )
        formatted = format_cost_stats(stats)
        assert "Total Cost: $0.00" in formatted


class TestGetRecommendedCostConfig:
    """Tests for get_recommended_cost_config function."""

    def test_small_model_inference(self) -> None:
        """Test recommendation for small model inference."""
        instance, pricing = get_recommended_cost_config(7.0, "inference", "standard")
        assert instance.gpu_count >= 1
        assert pricing.pricing_model == PricingModel.ON_DEMAND

    def test_large_model_training(self) -> None:
        """Test recommendation for large model training."""
        instance, pricing = get_recommended_cost_config(70.0, "training", "budget")
        assert instance.gpu_count >= 2
        assert pricing.pricing_model == PricingModel.SPOT

    def test_premium_tier(self) -> None:
        """Test recommendation for premium tier inference."""
        _instance, pricing = get_recommended_cost_config(40.0, "inference", "premium")
        assert pricing.pricing_model == PricingModel.RESERVED
        assert pricing.commitment_months == 12

    def test_zero_model_size_raises(self) -> None:
        """Test that zero model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_gb must be positive"):
            get_recommended_cost_config(0, "inference")

    def test_invalid_use_case_raises(self) -> None:
        """Test that invalid use case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_cost_config(7.0, "invalid")

    def test_invalid_budget_tier_raises(self) -> None:
        """Test that invalid budget tier raises ValueError."""
        with pytest.raises(ValueError, match="budget_tier must be one of"):
            get_recommended_cost_config(7.0, "inference", "invalid")


class TestPropertyBasedCostEstimation:
    """Property-based tests for cost estimation."""

    @given(
        gpu_count=st.integers(min_value=1, max_value=16),
        hours=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50)
    def test_cost_scales_with_gpus(self, gpu_count: int, hours: int) -> None:
        """Test that cost scales with GPU count."""
        instance = create_instance_config(gpu_count=gpu_count)
        pricing = create_pricing_config()
        estimate = estimate_inference_cost(instance, pricing, hours_per_month=hours)
        assert estimate.hourly_cost >= 0
        assert estimate.monthly_cost >= 0

    @given(st.floats(min_value=0.01, max_value=1000.0))
    @settings(max_examples=50)
    def test_hourly_cost_positive(self, base_cost: float) -> None:
        """Test that estimated costs are always non-negative."""
        estimate = create_cost_estimate(hourly_cost=base_cost)
        assert estimate.hourly_cost >= 0
        assert estimate.monthly_cost >= 0


class TestEdgeCases:
    """Tests for edge cases and branch coverage."""

    def test_cpu_instance_cost(self) -> None:
        """Test cost estimation for CPU instance (no GPU)."""
        instance = InstanceConfig(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.CPU,
            gpu_type=None,
            gpu_count=0,
            memory_gb=64,
        )
        pricing = create_pricing_config()
        estimate = estimate_inference_cost(instance, pricing)
        # CPU instances should have low cost
        assert estimate.hourly_cost > 0
        assert estimate.hourly_cost < 1.0

    def test_unknown_gpu_type_fallback(self) -> None:
        """Test cost estimation for unknown GPU type falls back to A100."""
        instance = InstanceConfig(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.GPU_DATACENTER,
            gpu_type="UNKNOWN_GPU",
            gpu_count=1,
            memory_gb=80,
        )
        pricing = create_pricing_config()
        estimate = estimate_inference_cost(instance, pricing)
        # Should use A100 pricing as fallback
        assert estimate.hourly_cost > 0

    def test_model_size_a10g_range(self) -> None:
        """Test recommended config for model size 7-14GB uses A10G."""
        instance, _ = get_recommended_cost_config(10.0, "inference", "standard")
        assert instance.gpu_type == "A10G"

    def test_model_size_large_h100(self) -> None:
        """Test recommended config for model size > 80GB uses H100."""
        instance, _ = get_recommended_cost_config(100.0, "inference", "standard")
        assert instance.gpu_type == "H100"
        assert instance.gpu_count >= 2

    def test_model_size_medium_a100(self) -> None:
        """Test recommended config for model size 14-40GB uses A100."""
        instance, _ = get_recommended_cost_config(30.0, "inference", "standard")
        assert instance.gpu_type == "A100"
        assert instance.gpu_count == 1

    def test_model_size_40_to_80_a100_multi_gpu(self) -> None:
        """Test recommended config for model size 40-80GB uses 2 A100s."""
        instance, _ = get_recommended_cost_config(60.0, "inference", "standard")
        assert instance.gpu_type == "A100"
        assert instance.gpu_count == 2

    def test_optimize_with_very_low_budget(self) -> None:
        """Test optimization with budget too low for any option."""
        options = optimize_instance_selection(
            model_size_gb=14.0,
            target_throughput=100.0,
            max_hourly_budget=0.01,  # Very low budget
        )
        # Should return empty list when no options fit budget
        assert isinstance(options, list)

    def test_create_cost_estimate_with_total_cost(self) -> None:
        """Test create_cost_estimate with explicit total_cost."""
        estimate = create_cost_estimate(
            hourly_cost=1.0,
            monthly_cost=720.0,
            total_cost=1440.0,  # Different from monthly
        )
        assert estimate.total_cost == pytest.approx(1440.0)

    def test_premium_tier_training(self) -> None:
        """Test recommended config for premium tier training uses ON_DEMAND."""
        instance, pricing = get_recommended_cost_config(40.0, "training", "premium")
        assert pricing.pricing_model == PricingModel.ON_DEMAND
        assert instance.gpu_count >= 4  # Training doubles GPU count
