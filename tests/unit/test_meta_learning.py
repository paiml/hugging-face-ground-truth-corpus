"""Tests for training.meta_learning module."""

from __future__ import annotations

import pytest

from hf_gtc.training.meta_learning import (
    VALID_ADAPTATION_STRATEGIES,
    VALID_META_LEARNING_METHODS,
    VALID_TASK_DISTRIBUTIONS,
    AdaptationStrategy,
    EpisodeConfig,
    MAMLConfig,
    MetaLearningConfig,
    MetaLearningMethod,
    MetaLearningStats,
    ProtoNetConfig,
    TaskDistribution,
    calculate_prototypes,
    compute_episode_accuracy,
    create_episode_config,
    create_maml_config,
    create_meta_learning_config,
    create_meta_learning_stats,
    create_protonet_config,
    estimate_adaptation_cost,
    evaluate_generalization,
    format_meta_learning_stats,
    get_adaptation_strategy,
    get_meta_learning_method,
    get_recommended_meta_learning_config,
    get_task_distribution,
    list_adaptation_strategies,
    list_meta_learning_methods,
    list_task_distributions,
    validate_episode_config,
    validate_maml_config,
    validate_meta_learning_config,
    validate_meta_learning_stats,
    validate_protonet_config,
)


class TestMetaLearningMethod:
    """Tests for MetaLearningMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in MetaLearningMethod:
            assert isinstance(method.value, str)

    def test_maml_value(self) -> None:
        """MAML has correct value."""
        assert MetaLearningMethod.MAML.value == "maml"

    def test_protonet_value(self) -> None:
        """PROTONET has correct value."""
        assert MetaLearningMethod.PROTONET.value == "protonet"

    def test_matching_net_value(self) -> None:
        """MATCHING_NET has correct value."""
        assert MetaLearningMethod.MATCHING_NET.value == "matching_net"

    def test_relation_net_value(self) -> None:
        """RELATION_NET has correct value."""
        assert MetaLearningMethod.RELATION_NET.value == "relation_net"

    def test_reptile_value(self) -> None:
        """REPTILE has correct value."""
        assert MetaLearningMethod.REPTILE.value == "reptile"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_META_LEARNING_METHODS is a frozenset."""
        assert isinstance(VALID_META_LEARNING_METHODS, frozenset)
        assert len(VALID_META_LEARNING_METHODS) == 5


class TestTaskDistribution:
    """Tests for TaskDistribution enum."""

    def test_all_distributions_have_values(self) -> None:
        """All distributions have string values."""
        for dist in TaskDistribution:
            assert isinstance(dist.value, str)

    def test_uniform_value(self) -> None:
        """UNIFORM has correct value."""
        assert TaskDistribution.UNIFORM.value == "uniform"

    def test_weighted_value(self) -> None:
        """WEIGHTED has correct value."""
        assert TaskDistribution.WEIGHTED.value == "weighted"

    def test_curriculum_value(self) -> None:
        """CURRICULUM has correct value."""
        assert TaskDistribution.CURRICULUM.value == "curriculum"

    def test_valid_distributions_frozenset(self) -> None:
        """VALID_TASK_DISTRIBUTIONS is a frozenset."""
        assert isinstance(VALID_TASK_DISTRIBUTIONS, frozenset)
        assert len(VALID_TASK_DISTRIBUTIONS) == 3


class TestAdaptationStrategy:
    """Tests for AdaptationStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in AdaptationStrategy:
            assert isinstance(strategy.value, str)

    def test_gradient_value(self) -> None:
        """GRADIENT has correct value."""
        assert AdaptationStrategy.GRADIENT.value == "gradient"

    def test_metric_value(self) -> None:
        """METRIC has correct value."""
        assert AdaptationStrategy.METRIC.value == "metric"

    def test_hybrid_value(self) -> None:
        """HYBRID has correct value."""
        assert AdaptationStrategy.HYBRID.value == "hybrid"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_ADAPTATION_STRATEGIES is a frozenset."""
        assert isinstance(VALID_ADAPTATION_STRATEGIES, frozenset)
        assert len(VALID_ADAPTATION_STRATEGIES) == 3


class TestMAMLConfig:
    """Tests for MAMLConfig dataclass."""

    def test_create_config(self) -> None:
        """Create MAML config."""
        config = MAMLConfig(
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=5,
            first_order=False,
        )
        assert config.inner_lr == 0.01
        assert config.outer_lr == 0.001
        assert config.inner_steps == 5
        assert config.first_order is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = MAMLConfig(0.01, 0.001, 5, False)
        with pytest.raises(AttributeError):
            config.inner_lr = 0.02  # type: ignore[misc]


class TestProtoNetConfig:
    """Tests for ProtoNetConfig dataclass."""

    def test_create_config(self) -> None:
        """Create ProtoNet config."""
        config = ProtoNetConfig(
            distance_metric="euclidean",
            embedding_dim=64,
            normalize=True,
        )
        assert config.distance_metric == "euclidean"
        assert config.embedding_dim == 64
        assert config.normalize is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ProtoNetConfig("euclidean", 64, True)
        with pytest.raises(AttributeError):
            config.embedding_dim = 128  # type: ignore[misc]


class TestEpisodeConfig:
    """Tests for EpisodeConfig dataclass."""

    def test_create_config(self) -> None:
        """Create episode config."""
        config = EpisodeConfig(
            n_way=5,
            k_shot=1,
            query_size=15,
        )
        assert config.n_way == 5
        assert config.k_shot == 1
        assert config.query_size == 15

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = EpisodeConfig(5, 1, 15)
        with pytest.raises(AttributeError):
            config.n_way = 10  # type: ignore[misc]


class TestMetaLearningConfig:
    """Tests for MetaLearningConfig dataclass."""

    def test_create_config(self) -> None:
        """Create meta-learning config."""
        episode = EpisodeConfig(5, 1, 15)
        maml = MAMLConfig(0.01, 0.001, 5, False)
        config = MetaLearningConfig(
            method=MetaLearningMethod.MAML,
            maml_config=maml,
            protonet_config=None,
            episode_config=episode,
        )
        assert config.method == MetaLearningMethod.MAML
        assert config.maml_config is not None
        assert config.protonet_config is None
        assert config.episode_config.n_way == 5

    def test_config_with_protonet(self) -> None:
        """Create config with ProtoNet."""
        episode = EpisodeConfig(5, 1, 15)
        protonet = ProtoNetConfig("euclidean", 64, True)
        config = MetaLearningConfig(
            method=MetaLearningMethod.PROTONET,
            maml_config=None,
            protonet_config=protonet,
            episode_config=episode,
        )
        assert config.method == MetaLearningMethod.PROTONET
        assert config.protonet_config is not None

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        episode = EpisodeConfig(5, 1, 15)
        config = MetaLearningConfig(MetaLearningMethod.MAML, None, None, episode)
        with pytest.raises(AttributeError):
            config.method = MetaLearningMethod.PROTONET  # type: ignore[misc]


class TestMetaLearningStats:
    """Tests for MetaLearningStats dataclass."""

    def test_create_stats(self) -> None:
        """Create meta-learning stats."""
        stats = MetaLearningStats(
            meta_train_accuracy=0.85,
            meta_test_accuracy=0.75,
            adaptation_steps=5.0,
            generalization_gap=0.10,
        )
        assert stats.meta_train_accuracy == 0.85
        assert stats.meta_test_accuracy == 0.75
        assert stats.adaptation_steps == 5.0
        assert stats.generalization_gap == 0.10

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = MetaLearningStats(0.85, 0.75, 5.0, 0.10)
        with pytest.raises(AttributeError):
            stats.meta_train_accuracy = 0.9  # type: ignore[misc]


class TestValidateMAMLConfig:
    """Tests for validate_maml_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = MAMLConfig(0.01, 0.001, 5, False)
        validate_maml_config(config)

    def test_negative_inner_lr_raises(self) -> None:
        """Negative inner_lr raises ValueError."""
        config = MAMLConfig(-0.01, 0.001, 5, False)
        with pytest.raises(ValueError, match="inner_lr must be positive"):
            validate_maml_config(config)

    def test_zero_inner_lr_raises(self) -> None:
        """Zero inner_lr raises ValueError."""
        config = MAMLConfig(0.0, 0.001, 5, False)
        with pytest.raises(ValueError, match="inner_lr must be positive"):
            validate_maml_config(config)

    def test_negative_outer_lr_raises(self) -> None:
        """Negative outer_lr raises ValueError."""
        config = MAMLConfig(0.01, -0.001, 5, False)
        with pytest.raises(ValueError, match="outer_lr must be positive"):
            validate_maml_config(config)

    def test_zero_outer_lr_raises(self) -> None:
        """Zero outer_lr raises ValueError."""
        config = MAMLConfig(0.01, 0.0, 5, False)
        with pytest.raises(ValueError, match="outer_lr must be positive"):
            validate_maml_config(config)

    def test_zero_inner_steps_raises(self) -> None:
        """Zero inner_steps raises ValueError."""
        config = MAMLConfig(0.01, 0.001, 0, False)
        with pytest.raises(ValueError, match="inner_steps must be positive"):
            validate_maml_config(config)

    def test_negative_inner_steps_raises(self) -> None:
        """Negative inner_steps raises ValueError."""
        config = MAMLConfig(0.01, 0.001, -1, False)
        with pytest.raises(ValueError, match="inner_steps must be positive"):
            validate_maml_config(config)


class TestValidateProtoNetConfig:
    """Tests for validate_protonet_config function."""

    def test_valid_euclidean_config(self) -> None:
        """Valid euclidean config passes validation."""
        config = ProtoNetConfig("euclidean", 64, True)
        validate_protonet_config(config)

    def test_valid_cosine_config(self) -> None:
        """Valid cosine config passes validation."""
        config = ProtoNetConfig("cosine", 64, True)
        validate_protonet_config(config)

    def test_invalid_metric_raises(self) -> None:
        """Invalid distance_metric raises ValueError."""
        config = ProtoNetConfig("invalid", 64, True)
        with pytest.raises(ValueError, match="distance_metric"):
            validate_protonet_config(config)

    def test_zero_embedding_dim_raises(self) -> None:
        """Zero embedding_dim raises ValueError."""
        config = ProtoNetConfig("euclidean", 0, True)
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            validate_protonet_config(config)

    def test_negative_embedding_dim_raises(self) -> None:
        """Negative embedding_dim raises ValueError."""
        config = ProtoNetConfig("euclidean", -64, True)
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            validate_protonet_config(config)


class TestValidateEpisodeConfig:
    """Tests for validate_episode_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = EpisodeConfig(5, 1, 15)
        validate_episode_config(config)

    def test_zero_n_way_raises(self) -> None:
        """Zero n_way raises ValueError."""
        config = EpisodeConfig(0, 1, 15)
        with pytest.raises(ValueError, match="n_way must be positive"):
            validate_episode_config(config)

    def test_negative_n_way_raises(self) -> None:
        """Negative n_way raises ValueError."""
        config = EpisodeConfig(-5, 1, 15)
        with pytest.raises(ValueError, match="n_way must be positive"):
            validate_episode_config(config)

    def test_zero_k_shot_raises(self) -> None:
        """Zero k_shot raises ValueError."""
        config = EpisodeConfig(5, 0, 15)
        with pytest.raises(ValueError, match="k_shot must be positive"):
            validate_episode_config(config)

    def test_negative_k_shot_raises(self) -> None:
        """Negative k_shot raises ValueError."""
        config = EpisodeConfig(5, -1, 15)
        with pytest.raises(ValueError, match="k_shot must be positive"):
            validate_episode_config(config)

    def test_zero_query_size_raises(self) -> None:
        """Zero query_size raises ValueError."""
        config = EpisodeConfig(5, 1, 0)
        with pytest.raises(ValueError, match="query_size must be positive"):
            validate_episode_config(config)

    def test_negative_query_size_raises(self) -> None:
        """Negative query_size raises ValueError."""
        config = EpisodeConfig(5, 1, -15)
        with pytest.raises(ValueError, match="query_size must be positive"):
            validate_episode_config(config)


class TestValidateMetaLearningConfig:
    """Tests for validate_meta_learning_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        episode = EpisodeConfig(5, 1, 15)
        maml = MAMLConfig(0.01, 0.001, 5, False)
        config = MetaLearningConfig(MetaLearningMethod.MAML, maml, None, episode)
        validate_meta_learning_config(config)

    def test_invalid_episode_raises(self) -> None:
        """Invalid episode_config raises ValueError."""
        bad_episode = EpisodeConfig(-1, 1, 15)
        config = MetaLearningConfig(
            MetaLearningMethod.PROTONET, None, None, bad_episode
        )
        with pytest.raises(ValueError, match="n_way must be positive"):
            validate_meta_learning_config(config)

    def test_invalid_maml_config_raises(self) -> None:
        """Invalid maml_config raises ValueError."""
        episode = EpisodeConfig(5, 1, 15)
        bad_maml = MAMLConfig(-0.01, 0.001, 5, False)
        config = MetaLearningConfig(MetaLearningMethod.MAML, bad_maml, None, episode)
        with pytest.raises(ValueError, match="inner_lr must be positive"):
            validate_meta_learning_config(config)

    def test_invalid_protonet_config_raises(self) -> None:
        """Invalid protonet_config raises ValueError."""
        episode = EpisodeConfig(5, 1, 15)
        bad_protonet = ProtoNetConfig("invalid", 64, True)
        config = MetaLearningConfig(
            MetaLearningMethod.PROTONET, None, bad_protonet, episode
        )
        with pytest.raises(ValueError, match="distance_metric"):
            validate_meta_learning_config(config)


class TestValidateMetaLearningStats:
    """Tests for validate_meta_learning_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = MetaLearningStats(0.85, 0.75, 5.0, 0.10)
        validate_meta_learning_stats(stats)

    def test_train_accuracy_above_one_raises(self) -> None:
        """meta_train_accuracy > 1 raises ValueError."""
        stats = MetaLearningStats(1.5, 0.75, 5.0, 0.10)
        with pytest.raises(ValueError, match="meta_train_accuracy must be between"):
            validate_meta_learning_stats(stats)

    def test_train_accuracy_below_zero_raises(self) -> None:
        """meta_train_accuracy < 0 raises ValueError."""
        stats = MetaLearningStats(-0.1, 0.75, 5.0, 0.10)
        with pytest.raises(ValueError, match="meta_train_accuracy must be between"):
            validate_meta_learning_stats(stats)

    def test_test_accuracy_above_one_raises(self) -> None:
        """meta_test_accuracy > 1 raises ValueError."""
        stats = MetaLearningStats(0.85, 1.5, 5.0, 0.10)
        with pytest.raises(ValueError, match="meta_test_accuracy must be between"):
            validate_meta_learning_stats(stats)

    def test_test_accuracy_below_zero_raises(self) -> None:
        """meta_test_accuracy < 0 raises ValueError."""
        stats = MetaLearningStats(0.85, -0.1, 5.0, 0.10)
        with pytest.raises(ValueError, match="meta_test_accuracy must be between"):
            validate_meta_learning_stats(stats)

    def test_negative_adaptation_steps_raises(self) -> None:
        """Negative adaptation_steps raises ValueError."""
        stats = MetaLearningStats(0.85, 0.75, -5.0, 0.10)
        with pytest.raises(ValueError, match="adaptation_steps must be non-negative"):
            validate_meta_learning_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        stats = MetaLearningStats(0.0, 0.0, 0.0, 0.0)
        validate_meta_learning_stats(stats)


class TestCreateMAMLConfig:
    """Tests for create_maml_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_maml_config()
        assert config.inner_lr == 0.01
        assert config.outer_lr == 0.001
        assert config.inner_steps == 5
        assert config.first_order is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_maml_config(
            inner_lr=0.05,
            outer_lr=0.01,
            inner_steps=10,
            first_order=True,
        )
        assert config.inner_lr == 0.05
        assert config.outer_lr == 0.01
        assert config.inner_steps == 10
        assert config.first_order is True

    def test_negative_inner_lr_raises(self) -> None:
        """Negative inner_lr raises ValueError."""
        with pytest.raises(ValueError, match="inner_lr must be positive"):
            create_maml_config(inner_lr=-0.01)

    def test_zero_inner_steps_raises(self) -> None:
        """Zero inner_steps raises ValueError."""
        with pytest.raises(ValueError, match="inner_steps must be positive"):
            create_maml_config(inner_steps=0)


class TestCreateProtoNetConfig:
    """Tests for create_protonet_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_protonet_config()
        assert config.distance_metric == "euclidean"
        assert config.embedding_dim == 64
        assert config.normalize is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_protonet_config(
            distance_metric="cosine",
            embedding_dim=128,
            normalize=False,
        )
        assert config.distance_metric == "cosine"
        assert config.embedding_dim == 128
        assert config.normalize is False

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="distance_metric"):
            create_protonet_config(distance_metric="invalid")

    def test_zero_embedding_dim_raises(self) -> None:
        """Zero embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            create_protonet_config(embedding_dim=0)


class TestCreateEpisodeConfig:
    """Tests for create_episode_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_episode_config()
        assert config.n_way == 5
        assert config.k_shot == 1
        assert config.query_size == 15

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_episode_config(
            n_way=10,
            k_shot=5,
            query_size=20,
        )
        assert config.n_way == 10
        assert config.k_shot == 5
        assert config.query_size == 20

    def test_zero_n_way_raises(self) -> None:
        """Zero n_way raises ValueError."""
        with pytest.raises(ValueError, match="n_way must be positive"):
            create_episode_config(n_way=0)

    def test_zero_k_shot_raises(self) -> None:
        """Zero k_shot raises ValueError."""
        with pytest.raises(ValueError, match="k_shot must be positive"):
            create_episode_config(k_shot=0)


class TestCreateMetaLearningConfig:
    """Tests for create_meta_learning_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_meta_learning_config()
        assert config.method == MetaLearningMethod.MAML
        assert config.maml_config is not None
        assert config.episode_config.n_way == 5

    def test_custom_maml_config(self) -> None:
        """Create custom MAML config."""
        maml = create_maml_config(inner_lr=0.05)
        config = create_meta_learning_config(
            method="maml",
            maml_config=maml,
        )
        assert config.method == MetaLearningMethod.MAML
        assert config.maml_config is not None
        assert config.maml_config.inner_lr == 0.05

    def test_protonet_config(self) -> None:
        """Create ProtoNet config."""
        config = create_meta_learning_config(method="protonet")
        assert config.method == MetaLearningMethod.PROTONET
        assert config.protonet_config is not None

    def test_with_enum_method(self) -> None:
        """Create with enum method."""
        config = create_meta_learning_config(method=MetaLearningMethod.REPTILE)
        assert config.method == MetaLearningMethod.REPTILE

    def test_custom_episode_config(self) -> None:
        """Create with custom episode config."""
        episode = create_episode_config(n_way=10, k_shot=5)
        config = create_meta_learning_config(episode_config=episode)
        assert config.episode_config.n_way == 10
        assert config.episode_config.k_shot == 5

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_meta_learning_config(method="invalid")

    @pytest.mark.parametrize(
        "method",
        ["maml", "protonet", "matching_net", "relation_net", "reptile"],
    )
    def test_all_methods(self, method: str) -> None:
        """Test all meta-learning methods."""
        config = create_meta_learning_config(method=method)
        assert config.method.value == method


class TestCreateMetaLearningStats:
    """Tests for create_meta_learning_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_meta_learning_stats()
        assert stats.meta_train_accuracy == 0.0
        assert stats.meta_test_accuracy == 0.0
        assert stats.adaptation_steps == 0.0
        assert stats.generalization_gap == 0.0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_meta_learning_stats(
            meta_train_accuracy=0.85,
            meta_test_accuracy=0.75,
            adaptation_steps=5.0,
            generalization_gap=0.10,
        )
        assert stats.meta_train_accuracy == 0.85
        assert stats.meta_test_accuracy == 0.75

    def test_train_accuracy_out_of_range_raises(self) -> None:
        """Train accuracy > 1 raises ValueError."""
        with pytest.raises(ValueError, match="meta_train_accuracy must be between"):
            create_meta_learning_stats(meta_train_accuracy=1.5)

    def test_negative_adaptation_steps_raises(self) -> None:
        """Negative adaptation_steps raises ValueError."""
        with pytest.raises(ValueError, match="adaptation_steps must be non-negative"):
            create_meta_learning_stats(adaptation_steps=-1.0)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_meta_learning_methods_sorted(self) -> None:
        """Returns sorted list."""
        methods = list_meta_learning_methods()
        assert methods == sorted(methods)
        assert "maml" in methods
        assert "protonet" in methods

    def test_list_task_distributions_sorted(self) -> None:
        """Returns sorted list."""
        distributions = list_task_distributions()
        assert distributions == sorted(distributions)
        assert "uniform" in distributions
        assert "curriculum" in distributions

    def test_list_adaptation_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_adaptation_strategies()
        assert strategies == sorted(strategies)
        assert "gradient" in strategies
        assert "metric" in strategies


class TestGetMetaLearningMethod:
    """Tests for get_meta_learning_method function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("maml", MetaLearningMethod.MAML),
            ("protonet", MetaLearningMethod.PROTONET),
            ("matching_net", MetaLearningMethod.MATCHING_NET),
            ("relation_net", MetaLearningMethod.RELATION_NET),
            ("reptile", MetaLearningMethod.REPTILE),
        ],
    )
    def test_all_methods(self, name: str, expected: MetaLearningMethod) -> None:
        """Test all valid methods."""
        assert get_meta_learning_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_meta_learning_method("invalid")


class TestGetTaskDistribution:
    """Tests for get_task_distribution function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("uniform", TaskDistribution.UNIFORM),
            ("weighted", TaskDistribution.WEIGHTED),
            ("curriculum", TaskDistribution.CURRICULUM),
        ],
    )
    def test_all_distributions(self, name: str, expected: TaskDistribution) -> None:
        """Test all valid distributions."""
        assert get_task_distribution(name) == expected

    def test_invalid_distribution_raises(self) -> None:
        """Invalid distribution raises ValueError."""
        with pytest.raises(ValueError, match="Invalid task_distribution"):
            get_task_distribution("invalid")


class TestGetAdaptationStrategy:
    """Tests for get_adaptation_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("gradient", AdaptationStrategy.GRADIENT),
            ("metric", AdaptationStrategy.METRIC),
            ("hybrid", AdaptationStrategy.HYBRID),
        ],
    )
    def test_all_strategies(self, name: str, expected: AdaptationStrategy) -> None:
        """Test all valid strategies."""
        assert get_adaptation_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="adaptation_strategy must be one of"):
            get_adaptation_strategy("invalid")


class TestCalculatePrototypes:
    """Tests for calculate_prototypes function."""

    def test_basic_prototypes(self) -> None:
        """Calculate basic prototypes."""
        embeddings = ((1.0, 0.0), (1.2, 0.1), (0.0, 1.0), (0.1, 1.1))
        labels = (0, 0, 1, 1)
        prototypes = calculate_prototypes(embeddings, labels, 2)
        assert len(prototypes) == 2
        # Class 0: mean of (1.0, 0.0) and (1.2, 0.1) = (1.1, 0.05)
        assert prototypes[0] == pytest.approx((1.1, 0.05))
        # Class 1: mean of (0.0, 1.0) and (0.1, 1.1) = (0.05, 1.05)
        assert prototypes[1] == pytest.approx((0.05, 1.05))

    def test_single_sample_per_class(self) -> None:
        """Prototype with single sample equals that sample."""
        embeddings = ((1.0, 2.0), (3.0, 4.0))
        labels = (0, 1)
        prototypes = calculate_prototypes(embeddings, labels, 2)
        assert prototypes[0] == (1.0, 2.0)
        assert prototypes[1] == (3.0, 4.0)

    def test_empty_class_returns_zeros(self) -> None:
        """Empty class gets zero prototype."""
        embeddings = ((1.0, 2.0),)
        labels = (0,)
        prototypes = calculate_prototypes(embeddings, labels, 2)
        assert prototypes[0] == (1.0, 2.0)
        assert prototypes[1] == (0.0, 0.0)

    def test_empty_embeddings_raises(self) -> None:
        """Empty embeddings raises ValueError."""
        with pytest.raises(ValueError, match="embeddings cannot be empty"):
            calculate_prototypes((), (), 2)

    def test_empty_labels_raises(self) -> None:
        """Empty labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be empty"):
            calculate_prototypes(((1.0,),), (), 2)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_prototypes(((1.0,),), (0, 1), 2)

    def test_zero_n_classes_raises(self) -> None:
        """Zero n_classes raises ValueError."""
        with pytest.raises(ValueError, match="n_classes must be positive"):
            calculate_prototypes(((1.0,),), (0,), 0)

    def test_invalid_label_raises(self) -> None:
        """Invalid label raises ValueError."""
        with pytest.raises(ValueError, match="label 5 out of range"):
            calculate_prototypes(((1.0,),), (5,), 2)


class TestComputeEpisodeAccuracy:
    """Tests for compute_episode_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        """Perfect classification gives accuracy 1.0."""
        prototypes = ((1.0, 0.0), (0.0, 1.0))
        query_embeddings = ((0.9, 0.1), (0.1, 0.9), (0.8, 0.2))
        query_labels = (0, 1, 0)
        acc = compute_episode_accuracy(query_embeddings, query_labels, prototypes)
        assert acc == 1.0

    def test_zero_accuracy(self) -> None:
        """All wrong predictions give accuracy 0.0."""
        prototypes = ((1.0, 0.0), (0.0, 1.0))
        query_embeddings = ((0.1, 0.9), (0.9, 0.1))
        query_labels = (0, 1)  # Swapped labels
        acc = compute_episode_accuracy(query_embeddings, query_labels, prototypes)
        assert acc == 0.0

    def test_partial_accuracy(self) -> None:
        """Partial correct predictions."""
        prototypes = ((1.0, 0.0), (0.0, 1.0))
        query_embeddings = ((0.9, 0.1), (0.9, 0.1))  # Both near prototype 0
        query_labels = (0, 1)  # One correct, one wrong
        acc = compute_episode_accuracy(query_embeddings, query_labels, prototypes)
        assert acc == 0.5

    def test_cosine_distance(self) -> None:
        """Test with cosine distance."""
        prototypes = ((1.0, 0.0), (0.0, 1.0))
        query_embeddings = ((0.9, 0.1), (0.1, 0.9))
        query_labels = (0, 1)
        acc = compute_episode_accuracy(
            query_embeddings, query_labels, prototypes, distance_metric="cosine"
        )
        assert acc == 1.0

    def test_empty_query_embeddings_raises(self) -> None:
        """Empty query embeddings raises ValueError."""
        with pytest.raises(ValueError, match="query_embeddings cannot be empty"):
            compute_episode_accuracy((), (), ((1.0,),))

    def test_empty_query_labels_raises(self) -> None:
        """Empty query labels raises ValueError."""
        with pytest.raises(ValueError, match="query_labels cannot be empty"):
            compute_episode_accuracy(((1.0,),), (), ((1.0,),))

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            compute_episode_accuracy(((1.0,),), (0, 1), ((1.0,),))

    def test_empty_prototypes_raises(self) -> None:
        """Empty prototypes raises ValueError."""
        with pytest.raises(ValueError, match="prototypes cannot be empty"):
            compute_episode_accuracy(((1.0,),), (0,), ())

    def test_invalid_distance_metric_raises(self) -> None:
        """Invalid distance metric raises ValueError."""
        with pytest.raises(ValueError, match="distance_metric"):
            compute_episode_accuracy(
                ((1.0,),), (0,), ((1.0,),), distance_metric="invalid"
            )


class TestEstimateAdaptationCost:
    """Tests for estimate_adaptation_cost function."""

    def test_maml_cost(self) -> None:
        """Estimate MAML adaptation cost."""
        cost = estimate_adaptation_cost(
            method=MetaLearningMethod.MAML,
            inner_steps=5,
            n_way=5,
            k_shot=1,
            model_parameters=1_000_000,
        )
        assert "flops" in cost
        assert "memory_gb" in cost
        assert cost["flops"] > 0
        assert cost["memory_gb"] > 0

    def test_reptile_cost(self) -> None:
        """Estimate Reptile adaptation cost."""
        cost = estimate_adaptation_cost(
            method=MetaLearningMethod.REPTILE,
            inner_steps=5,
            n_way=5,
            k_shot=1,
            model_parameters=1_000_000,
        )
        assert cost["flops"] > 0

    def test_protonet_cost(self) -> None:
        """Estimate ProtoNet adaptation cost."""
        cost = estimate_adaptation_cost(
            method=MetaLearningMethod.PROTONET,
            inner_steps=5,
            n_way=5,
            k_shot=1,
            model_parameters=1_000_000,
        )
        assert cost["flops"] > 0

    def test_maml_more_expensive_than_protonet(self) -> None:
        """MAML should be more expensive than ProtoNet."""
        maml_cost = estimate_adaptation_cost(
            MetaLearningMethod.MAML, 5, 5, 1, 1_000_000
        )
        protonet_cost = estimate_adaptation_cost(
            MetaLearningMethod.PROTONET, 5, 5, 1, 1_000_000
        )
        assert maml_cost["flops"] > protonet_cost["flops"]
        assert maml_cost["memory_gb"] >= protonet_cost["memory_gb"]

    def test_zero_inner_steps_raises(self) -> None:
        """Zero inner_steps raises ValueError."""
        with pytest.raises(ValueError, match="inner_steps must be positive"):
            estimate_adaptation_cost(MetaLearningMethod.MAML, 0, 5, 1, 1_000_000)

    def test_zero_n_way_raises(self) -> None:
        """Zero n_way raises ValueError."""
        with pytest.raises(ValueError, match="n_way must be positive"):
            estimate_adaptation_cost(MetaLearningMethod.MAML, 5, 0, 1, 1_000_000)

    def test_zero_k_shot_raises(self) -> None:
        """Zero k_shot raises ValueError."""
        with pytest.raises(ValueError, match="k_shot must be positive"):
            estimate_adaptation_cost(MetaLearningMethod.MAML, 5, 5, 0, 1_000_000)

    def test_zero_model_parameters_raises(self) -> None:
        """Zero model_parameters raises ValueError."""
        with pytest.raises(ValueError, match="model_parameters must be positive"):
            estimate_adaptation_cost(MetaLearningMethod.MAML, 5, 5, 1, 0)


class TestEvaluateGeneralization:
    """Tests for evaluate_generalization function."""

    def test_basic_evaluation(self) -> None:
        """Evaluate basic generalization."""
        train_acc = (0.90, 0.85, 0.88)
        test_acc = (0.80, 0.75, 0.78)
        metrics = evaluate_generalization(train_acc, test_acc)
        assert "mean_train_accuracy" in metrics
        assert "mean_test_accuracy" in metrics
        assert "generalization_gap" in metrics
        assert "train_std" in metrics
        assert "test_std" in metrics

    def test_mean_accuracy(self) -> None:
        """Test mean accuracy calculation."""
        train_acc = (0.90, 0.80)
        test_acc = (0.70, 0.60)
        metrics = evaluate_generalization(train_acc, test_acc)
        assert metrics["mean_train_accuracy"] == pytest.approx(0.85)
        assert metrics["mean_test_accuracy"] == pytest.approx(0.65)

    def test_generalization_gap(self) -> None:
        """Test generalization gap calculation."""
        train_acc = (0.90, 0.80)
        test_acc = (0.70, 0.60)
        metrics = evaluate_generalization(train_acc, test_acc)
        assert metrics["generalization_gap"] == pytest.approx(0.20)

    def test_no_gap(self) -> None:
        """Test when train equals test."""
        train_acc = (0.80, 0.80)
        test_acc = (0.80, 0.80)
        metrics = evaluate_generalization(train_acc, test_acc)
        assert metrics["generalization_gap"] == pytest.approx(0.0)

    def test_negative_gap(self) -> None:
        """Test negative gap (test better than train)."""
        train_acc = (0.70,)
        test_acc = (0.80,)
        metrics = evaluate_generalization(train_acc, test_acc)
        assert metrics["generalization_gap"] < 0

    def test_empty_train_accuracies_raises(self) -> None:
        """Empty train_accuracies raises ValueError."""
        with pytest.raises(ValueError, match="train_accuracies cannot be empty"):
            evaluate_generalization((), (0.8,))

    def test_empty_test_accuracies_raises(self) -> None:
        """Empty test_accuracies raises ValueError."""
        with pytest.raises(ValueError, match="test_accuracies cannot be empty"):
            evaluate_generalization((0.8,), ())


class TestFormatMetaLearningStats:
    """Tests for format_meta_learning_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_meta_learning_stats(
            meta_train_accuracy=0.85,
            meta_test_accuracy=0.75,
            adaptation_steps=5.0,
            generalization_gap=0.10,
        )
        formatted = format_meta_learning_stats(stats)
        assert "Meta-Train Accuracy: 85.0%" in formatted
        assert "Meta-Test Accuracy: 75.0%" in formatted
        assert "Adaptation Steps: 5.0" in formatted
        assert "Generalization Gap: 10.0%" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_meta_learning_stats()
        formatted = format_meta_learning_stats(stats)
        assert "Meta-Train Accuracy:" in formatted
        assert "Meta-Test Accuracy:" in formatted
        assert "Adaptation Steps:" in formatted
        assert "Generalization Gap:" in formatted


class TestGetRecommendedMetaLearningConfig:
    """Tests for get_recommended_meta_learning_config function."""

    def test_classification_few_samples(self) -> None:
        """Classification with few samples recommends ProtoNet."""
        config = get_recommended_meta_learning_config("classification", 5)
        assert config.method == MetaLearningMethod.PROTONET

    def test_classification_many_samples(self) -> None:
        """Classification with many samples recommends MAML."""
        config = get_recommended_meta_learning_config("classification", 20)
        assert config.method == MetaLearningMethod.MAML

    def test_regression_recommends_reptile(self) -> None:
        """Regression recommends Reptile."""
        config = get_recommended_meta_learning_config("regression", 10)
        assert config.method == MetaLearningMethod.REPTILE

    def test_reinforcement_recommends_maml(self) -> None:
        """Reinforcement learning recommends MAML."""
        config = get_recommended_meta_learning_config("reinforcement", 10)
        assert config.method == MetaLearningMethod.MAML
        assert config.maml_config is not None
        assert config.maml_config.first_order is True

    def test_invalid_task_type_raises(self) -> None:
        """Invalid task_type raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_meta_learning_config("unknown", 10)

    def test_zero_samples_raises(self) -> None:
        """Zero samples raises ValueError."""
        with pytest.raises(ValueError, match="available_samples_per_class must be"):
            get_recommended_meta_learning_config("classification", 0)

    def test_negative_samples_raises(self) -> None:
        """Negative samples raises ValueError."""
        with pytest.raises(ValueError, match="available_samples_per_class must be"):
            get_recommended_meta_learning_config("classification", -5)

    @pytest.mark.parametrize(
        "task_type",
        ["classification", "regression", "reinforcement"],
    )
    def test_all_task_types(self, task_type: str) -> None:
        """Test all valid task types."""
        config = get_recommended_meta_learning_config(task_type, 10)
        assert config is not None
        assert config.episode_config is not None
