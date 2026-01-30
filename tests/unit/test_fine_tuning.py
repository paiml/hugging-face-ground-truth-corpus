"""Tests for fine-tuning functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hf_gtc.training.fine_tuning import (
    VALID_EVAL_STRATEGIES,
    VALID_SAVE_STRATEGIES,
    TrainingConfig,
    compute_num_training_steps,
    create_trainer,
    create_training_args,
    validate_training_config,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig(output_dir="/tmp/test")
        assert config.output_dir == "/tmp/test"
        assert config.num_epochs == 3
        assert config.batch_size == 8
        assert config.learning_rate == 5e-5
        assert config.weight_decay == 0.01
        assert config.warmup_ratio == 0.1
        assert config.eval_strategy == "epoch"
        assert config.save_strategy == "epoch"
        assert config.logging_steps == 100
        assert config.fp16 is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TrainingConfig(
            output_dir="/custom/path",
            num_epochs=10,
            batch_size=16,
            learning_rate=1e-4,
            weight_decay=0.05,
            warmup_ratio=0.2,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=50,
            fp16=True,
        )
        assert config.output_dir == "/custom/path"
        assert config.num_epochs == 10
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.05
        assert config.warmup_ratio == 0.2
        assert config.eval_strategy == "steps"
        assert config.save_strategy == "steps"
        assert config.logging_steps == 50
        assert config.fp16 is True

    def test_frozen(self) -> None:
        """Test that TrainingConfig is immutable."""
        config = TrainingConfig(output_dir="/tmp/test")
        with pytest.raises(AttributeError):
            config.output_dir = "/changed"  # type: ignore[misc]


class TestValidateTrainingConfig:
    """Tests for validate_training_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = TrainingConfig(output_dir="/tmp/test")
        validate_training_config(config)  # Should not raise

    def test_empty_output_dir(self) -> None:
        """Test validation fails for empty output_dir."""
        config = TrainingConfig(output_dir="")
        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            validate_training_config(config)

    def test_zero_num_epochs(self) -> None:
        """Test validation fails for zero num_epochs."""
        config = TrainingConfig(output_dir="/tmp/test", num_epochs=0)
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            validate_training_config(config)

    def test_negative_num_epochs(self) -> None:
        """Test validation fails for negative num_epochs."""
        config = TrainingConfig(output_dir="/tmp/test", num_epochs=-1)
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            validate_training_config(config)

    def test_zero_batch_size(self) -> None:
        """Test validation fails for zero batch_size."""
        config = TrainingConfig(output_dir="/tmp/test", batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_training_config(config)

    def test_negative_batch_size(self) -> None:
        """Test validation fails for negative batch_size."""
        config = TrainingConfig(output_dir="/tmp/test", batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_training_config(config)

    def test_zero_learning_rate(self) -> None:
        """Test validation fails for zero learning_rate."""
        config = TrainingConfig(output_dir="/tmp/test", learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_training_config(config)

    def test_negative_learning_rate(self) -> None:
        """Test validation fails for negative learning_rate."""
        config = TrainingConfig(output_dir="/tmp/test", learning_rate=-0.001)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_training_config(config)

    def test_negative_weight_decay(self) -> None:
        """Test validation fails for negative weight_decay."""
        config = TrainingConfig(output_dir="/tmp/test", weight_decay=-0.01)
        with pytest.raises(ValueError, match="weight_decay cannot be negative"):
            validate_training_config(config)

    def test_warmup_ratio_too_low(self) -> None:
        """Test validation fails for warmup_ratio below 0."""
        config = TrainingConfig(output_dir="/tmp/test", warmup_ratio=-0.1)
        with pytest.raises(ValueError, match="warmup_ratio must be between 0 and 1"):
            validate_training_config(config)

    def test_warmup_ratio_too_high(self) -> None:
        """Test validation fails for warmup_ratio above 1."""
        config = TrainingConfig(output_dir="/tmp/test", warmup_ratio=1.5)
        with pytest.raises(ValueError, match="warmup_ratio must be between 0 and 1"):
            validate_training_config(config)

    def test_invalid_eval_strategy(self) -> None:
        """Test validation fails for invalid eval_strategy."""
        config = TrainingConfig(output_dir="/tmp/test", eval_strategy="invalid")
        with pytest.raises(ValueError, match="eval_strategy must be one of"):
            validate_training_config(config)

    def test_invalid_save_strategy(self) -> None:
        """Test validation fails for invalid save_strategy."""
        config = TrainingConfig(output_dir="/tmp/test", save_strategy="invalid")
        with pytest.raises(ValueError, match="save_strategy must be one of"):
            validate_training_config(config)

    def test_zero_logging_steps(self) -> None:
        """Test validation fails for zero logging_steps."""
        config = TrainingConfig(output_dir="/tmp/test", logging_steps=0)
        with pytest.raises(ValueError, match="logging_steps must be positive"):
            validate_training_config(config)

    def test_negative_logging_steps(self) -> None:
        """Test validation fails for negative logging_steps."""
        config = TrainingConfig(output_dir="/tmp/test", logging_steps=-10)
        with pytest.raises(ValueError, match="logging_steps must be positive"):
            validate_training_config(config)


class TestValidStrategies:
    """Tests for valid strategy constants."""

    def test_eval_strategies_contents(self) -> None:
        """Test VALID_EVAL_STRATEGIES contains expected values."""
        assert "epoch" in VALID_EVAL_STRATEGIES
        assert "steps" in VALID_EVAL_STRATEGIES
        assert "no" in VALID_EVAL_STRATEGIES
        assert len(VALID_EVAL_STRATEGIES) == 3

    def test_save_strategies_contents(self) -> None:
        """Test VALID_SAVE_STRATEGIES contains expected values."""
        assert "epoch" in VALID_SAVE_STRATEGIES
        assert "steps" in VALID_SAVE_STRATEGIES
        assert "no" in VALID_SAVE_STRATEGIES
        assert len(VALID_SAVE_STRATEGIES) == 3


class TestCreateTrainingArgs:
    """Tests for create_training_args function."""

    def test_creates_training_args(self) -> None:
        """Test successful creation of TrainingArguments."""
        args = create_training_args("/tmp/test")
        assert args.output_dir == "/tmp/test"
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 8
        assert args.learning_rate == 5e-5

    def test_custom_parameters(self) -> None:
        """Test custom parameter values."""
        args = create_training_args(
            "/tmp/test",
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-4,
            fp16=True,
        )
        assert args.num_train_epochs == 5
        assert args.per_device_train_batch_size == 16
        assert args.learning_rate == 1e-4
        assert args.fp16 is True

    def test_empty_output_dir_raises_error(self) -> None:
        """Test that empty output_dir raises ValueError."""
        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            create_training_args("")

    def test_invalid_num_epochs_raises_error(self) -> None:
        """Test that invalid num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            create_training_args("/tmp/test", num_epochs=0)

    def test_invalid_batch_size_raises_error(self) -> None:
        """Test that invalid batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_training_args("/tmp/test", batch_size=0)


class TestCreateTrainer:
    """Tests for create_trainer function."""

    @patch("hf_gtc.training.fine_tuning._HFTrainer")
    def test_creates_trainer(self, mock_trainer_cls: MagicMock) -> None:
        """Test successful creation of Trainer."""
        model = MagicMock()
        args = MagicMock()
        dataset = MagicMock()

        trainer = create_trainer(model, args, dataset)

        mock_trainer_cls.assert_called_once()
        assert trainer is not None

    @patch("hf_gtc.training.fine_tuning._HFTrainer")
    def test_passes_all_parameters(self, mock_trainer_cls: MagicMock) -> None:
        """Test all parameters are passed to Trainer."""
        model = MagicMock()
        args = MagicMock()
        train_dataset = MagicMock()
        eval_dataset = MagicMock()
        tokenizer = MagicMock()
        compute_metrics = MagicMock()

        create_trainer(
            model,
            args,
            train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        mock_trainer_cls.assert_called_once_with(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    def test_none_model_raises_error(self) -> None:
        """Test that None model raises ValueError."""
        with pytest.raises(ValueError, match="model cannot be None"):
            create_trainer(None, MagicMock(), MagicMock())  # type: ignore[arg-type]

    def test_none_train_dataset_raises_error(self) -> None:
        """Test that None train_dataset raises ValueError."""
        with pytest.raises(ValueError, match="train_dataset cannot be None"):
            create_trainer(MagicMock(), MagicMock(), None)  # type: ignore[arg-type]


class TestComputeNumTrainingSteps:
    """Tests for compute_num_training_steps function."""

    def test_basic_computation(self) -> None:
        """Test basic training steps computation."""
        steps = compute_num_training_steps(1000, 8, 3)
        # 1000 / 8 = 125 steps per epoch, 125 * 3 = 375
        assert steps == 375

    def test_with_gradient_accumulation(self) -> None:
        """Test computation with gradient accumulation."""
        steps = compute_num_training_steps(1000, 8, 3, gradient_accumulation_steps=2)
        # 1000 / (8 * 2) = 62 steps per epoch, 62 * 3 = 186
        assert steps == 186

    def test_small_dataset(self) -> None:
        """Test computation with small dataset."""
        steps = compute_num_training_steps(100, 32, 1)
        # 100 / 32 = 3 steps per epoch (integer division)
        assert steps == 3

    def test_zero_num_samples_raises_error(self) -> None:
        """Test that zero num_samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            compute_num_training_steps(0, 8, 3)

    def test_negative_num_samples_raises_error(self) -> None:
        """Test that negative num_samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            compute_num_training_steps(-100, 8, 3)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            compute_num_training_steps(1000, 0, 3)

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            compute_num_training_steps(1000, -8, 3)

    def test_zero_num_epochs_raises_error(self) -> None:
        """Test that zero num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            compute_num_training_steps(1000, 8, 0)

    def test_negative_num_epochs_raises_error(self) -> None:
        """Test that negative num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            compute_num_training_steps(1000, 8, -1)

    def test_zero_gradient_accumulation_raises_error(self) -> None:
        """Test that zero gradient_accumulation_steps raises ValueError."""
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps must be positive"
        ):
            compute_num_training_steps(1000, 8, 3, gradient_accumulation_steps=0)

    def test_negative_gradient_accumulation_raises_error(self) -> None:
        """Test that negative gradient_accumulation_steps raises ValueError."""
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps must be positive"
        ):
            compute_num_training_steps(1000, 8, 3, gradient_accumulation_steps=-1)
