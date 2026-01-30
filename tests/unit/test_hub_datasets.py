"""Tests for hub datasets functionality."""

from __future__ import annotations

import pytest

from hf_gtc.hub.datasets import (
    VALID_FORMATS,
    VALID_SPLITS,
    VALID_STREAMING_MODES,
    DatasetConfig,
    DatasetFormat,
    DatasetStats,
    DownloadConfig,
    SplitType,
    StreamingMode,
    UploadConfig,
    calculate_dataset_hash,
    compare_dataset_schemas,
    create_dataset_config,
    create_dataset_stats,
    create_download_config,
    create_upload_config,
    estimate_download_size,
    format_dataset_stats,
    get_format,
    get_recommended_dataset_config,
    get_split,
    get_streaming_mode,
    list_formats,
    list_splits,
    list_streaming_modes,
    validate_dataset_config,
    validate_dataset_format,
    validate_dataset_stats,
    validate_download_config,
    validate_upload_config,
)


class TestDatasetFormatEnum:
    """Tests for DatasetFormat enum."""

    def test_format_values(self) -> None:
        """Test all format values exist."""
        assert DatasetFormat.PARQUET.value == "parquet"
        assert DatasetFormat.JSON.value == "json"
        assert DatasetFormat.CSV.value == "csv"
        assert DatasetFormat.ARROW.value == "arrow"
        assert DatasetFormat.WEBDATASET.value == "webdataset"

    def test_valid_formats_frozenset(self) -> None:
        """Test VALID_FORMATS contains all enum values."""
        assert "parquet" in VALID_FORMATS
        assert "json" in VALID_FORMATS
        assert "csv" in VALID_FORMATS
        assert "arrow" in VALID_FORMATS
        assert "webdataset" in VALID_FORMATS
        assert len(VALID_FORMATS) == 5


class TestSplitTypeEnum:
    """Tests for SplitType enum."""

    def test_split_values(self) -> None:
        """Test all split values exist."""
        assert SplitType.TRAIN.value == "train"
        assert SplitType.VALIDATION.value == "validation"
        assert SplitType.TEST.value == "test"
        assert SplitType.ALL.value == "all"

    def test_valid_splits_frozenset(self) -> None:
        """Test VALID_SPLITS contains all enum values."""
        assert "train" in VALID_SPLITS
        assert "validation" in VALID_SPLITS
        assert "test" in VALID_SPLITS
        assert "all" in VALID_SPLITS
        assert len(VALID_SPLITS) == 4


class TestStreamingModeEnum:
    """Tests for StreamingMode enum."""

    def test_streaming_values(self) -> None:
        """Test all streaming mode values exist."""
        assert StreamingMode.DISABLED.value == "disabled"
        assert StreamingMode.BASIC.value == "basic"
        assert StreamingMode.SHUFFLED.value == "shuffled"

    def test_valid_streaming_modes_frozenset(self) -> None:
        """Test VALID_STREAMING_MODES contains all enum values."""
        assert "disabled" in VALID_STREAMING_MODES
        assert "basic" in VALID_STREAMING_MODES
        assert "shuffled" in VALID_STREAMING_MODES
        assert len(VALID_STREAMING_MODES) == 3


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_dataset_config_creation(self) -> None:
        """Test creating DatasetConfig instance."""
        config = DatasetConfig(
            name="squad",
            subset="v1.1",
            split=SplitType.TRAIN,
            streaming=StreamingMode.DISABLED,
            trust_remote_code=False,
        )
        assert config.name == "squad"
        assert config.subset == "v1.1"
        assert config.split == SplitType.TRAIN
        assert config.streaming == StreamingMode.DISABLED
        assert config.trust_remote_code is False

    def test_dataset_config_frozen(self) -> None:
        """Test that DatasetConfig is immutable."""
        config = DatasetConfig(
            name="test",
            subset=None,
            split=SplitType.TRAIN,
            streaming=StreamingMode.DISABLED,
            trust_remote_code=False,
        )
        with pytest.raises(AttributeError):
            config.name = "changed"  # type: ignore[misc]


class TestDownloadConfig:
    """Tests for DownloadConfig dataclass."""

    def test_download_config_creation(self) -> None:
        """Test creating DownloadConfig instance."""
        config = DownloadConfig(
            cache_dir="/tmp/cache",
            force_download=True,
            resume_download=True,
            max_retries=5,
        )
        assert config.cache_dir == "/tmp/cache"
        assert config.force_download is True
        assert config.resume_download is True
        assert config.max_retries == 5

    def test_download_config_frozen(self) -> None:
        """Test that DownloadConfig is immutable."""
        config = DownloadConfig(
            cache_dir=None,
            force_download=False,
            resume_download=True,
            max_retries=3,
        )
        with pytest.raises(AttributeError):
            config.max_retries = 10  # type: ignore[misc]


class TestUploadConfig:
    """Tests for UploadConfig dataclass."""

    def test_upload_config_creation(self) -> None:
        """Test creating UploadConfig instance."""
        config = UploadConfig(
            repo_id="user/dataset",
            private=True,
            commit_message="Initial upload",
            create_pr=True,
        )
        assert config.repo_id == "user/dataset"
        assert config.private is True
        assert config.commit_message == "Initial upload"
        assert config.create_pr is True

    def test_upload_config_frozen(self) -> None:
        """Test that UploadConfig is immutable."""
        config = UploadConfig(
            repo_id="user/ds",
            private=False,
            commit_message="msg",
            create_pr=False,
        )
        with pytest.raises(AttributeError):
            config.private = True  # type: ignore[misc]


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_dataset_stats_creation(self) -> None:
        """Test creating DatasetStats instance."""
        stats = DatasetStats(
            num_rows=10000,
            num_columns=5,
            size_bytes=1048576,
            features={"text": "string", "label": "int64"},
        )
        assert stats.num_rows == 10000
        assert stats.num_columns == 5
        assert stats.size_bytes == 1048576
        assert stats.features["text"] == "string"

    def test_dataset_stats_frozen(self) -> None:
        """Test that DatasetStats is immutable."""
        stats = DatasetStats(
            num_rows=100,
            num_columns=2,
            size_bytes=1000,
            features={},
        )
        with pytest.raises(AttributeError):
            stats.num_rows = 200  # type: ignore[misc]


class TestValidateDatasetConfig:
    """Tests for validate_dataset_config function."""

    def test_valid_config(self) -> None:
        """Test validating a valid config."""
        config = DatasetConfig(
            "squad", None, SplitType.TRAIN, StreamingMode.DISABLED, False
        )
        validate_dataset_config(config)  # Should not raise

    def test_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        config = DatasetConfig(
            "", None, SplitType.TRAIN, StreamingMode.DISABLED, False
        )
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_dataset_config(config)


class TestValidateDownloadConfig:
    """Tests for validate_download_config function."""

    def test_valid_config(self) -> None:
        """Test validating a valid config."""
        config = DownloadConfig("/tmp", False, True, 3)
        validate_download_config(config)  # Should not raise

    def test_negative_retries_raises(self) -> None:
        """Test that negative max_retries raises ValueError."""
        config = DownloadConfig(None, False, True, -1)
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            validate_download_config(config)


class TestValidateUploadConfig:
    """Tests for validate_upload_config function."""

    def test_valid_config(self) -> None:
        """Test validating a valid config."""
        config = UploadConfig("user/ds", False, "message", False)
        validate_upload_config(config)  # Should not raise

    def test_empty_repo_id_raises(self) -> None:
        """Test that empty repo_id raises ValueError."""
        config = UploadConfig("", False, "message", False)
        with pytest.raises(ValueError, match="repo_id cannot be empty"):
            validate_upload_config(config)

    def test_empty_commit_message_raises(self) -> None:
        """Test that empty commit_message raises ValueError."""
        config = UploadConfig("user/ds", False, "", False)
        with pytest.raises(ValueError, match="commit_message cannot be empty"):
            validate_upload_config(config)


class TestValidateDatasetStats:
    """Tests for validate_dataset_stats function."""

    def test_valid_stats(self) -> None:
        """Test validating valid stats."""
        stats = DatasetStats(1000, 5, 10240, {"text": "string"})
        validate_dataset_stats(stats)  # Should not raise

    def test_negative_num_rows_raises(self) -> None:
        """Test that negative num_rows raises ValueError."""
        stats = DatasetStats(-1, 5, 10240, {})
        with pytest.raises(ValueError, match="num_rows must be non-negative"):
            validate_dataset_stats(stats)

    def test_negative_num_columns_raises(self) -> None:
        """Test that negative num_columns raises ValueError."""
        stats = DatasetStats(100, -1, 10240, {})
        with pytest.raises(ValueError, match="num_columns must be non-negative"):
            validate_dataset_stats(stats)

    def test_negative_size_bytes_raises(self) -> None:
        """Test that negative size_bytes raises ValueError."""
        stats = DatasetStats(100, 5, -1, {})
        with pytest.raises(ValueError, match="size_bytes must be non-negative"):
            validate_dataset_stats(stats)


class TestCreateDatasetConfig:
    """Tests for create_dataset_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with default values."""
        config = create_dataset_config("squad")
        assert config.name == "squad"
        assert config.subset is None
        assert config.split == SplitType.TRAIN
        assert config.streaming == StreamingMode.DISABLED
        assert config.trust_remote_code is False

    def test_create_with_all_params(self) -> None:
        """Test creating config with all parameters."""
        config = create_dataset_config(
            name="glue",
            subset="mrpc",
            split="validation",
            streaming="basic",
            trust_remote_code=True,
        )
        assert config.name == "glue"
        assert config.subset == "mrpc"
        assert config.split == SplitType.VALIDATION
        assert config.streaming == StreamingMode.BASIC
        assert config.trust_remote_code is True

    def test_create_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_dataset_config("")

    def test_create_invalid_split_raises(self) -> None:
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            create_dataset_config("ds", split="invalid")

    def test_create_invalid_streaming_raises(self) -> None:
        """Test that invalid streaming mode raises ValueError."""
        with pytest.raises(ValueError, match="streaming must be one of"):
            create_dataset_config("ds", streaming="invalid")

    def test_create_all_splits(self) -> None:
        """Test creating config with all valid splits."""
        for split in VALID_SPLITS:
            config = create_dataset_config("ds", split=split)
            assert config.split.value == split

    def test_create_all_streaming_modes(self) -> None:
        """Test creating config with all streaming modes."""
        for mode in VALID_STREAMING_MODES:
            config = create_dataset_config("ds", streaming=mode)
            assert config.streaming.value == mode


class TestCreateDownloadConfig:
    """Tests for create_download_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with default values."""
        config = create_download_config()
        assert config.cache_dir is None
        assert config.force_download is False
        assert config.resume_download is True
        assert config.max_retries == 3

    def test_create_with_all_params(self) -> None:
        """Test creating config with all parameters."""
        config = create_download_config(
            cache_dir="/data/cache",
            force_download=True,
            resume_download=False,
            max_retries=5,
        )
        assert config.cache_dir == "/data/cache"
        assert config.force_download is True
        assert config.resume_download is False
        assert config.max_retries == 5

    def test_create_negative_retries_raises(self) -> None:
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            create_download_config(max_retries=-1)


class TestCreateUploadConfig:
    """Tests for create_upload_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with default values."""
        config = create_upload_config("user/my-dataset")
        assert config.repo_id == "user/my-dataset"
        assert config.private is False
        assert config.commit_message == "Upload dataset"
        assert config.create_pr is False

    def test_create_with_all_params(self) -> None:
        """Test creating config with all parameters."""
        config = create_upload_config(
            repo_id="org/dataset",
            private=True,
            commit_message="Add v2 data",
            create_pr=True,
        )
        assert config.repo_id == "org/dataset"
        assert config.private is True
        assert config.commit_message == "Add v2 data"
        assert config.create_pr is True

    def test_create_empty_repo_id_raises(self) -> None:
        """Test that empty repo_id raises ValueError."""
        with pytest.raises(ValueError, match="repo_id cannot be empty"):
            create_upload_config("")


class TestCreateDatasetStats:
    """Tests for create_dataset_stats function."""

    def test_create_with_defaults(self) -> None:
        """Test creating stats with default values."""
        stats = create_dataset_stats()
        assert stats.num_rows == 0
        assert stats.num_columns == 0
        assert stats.size_bytes == 0
        assert stats.features == {}

    def test_create_with_all_params(self) -> None:
        """Test creating stats with all parameters."""
        stats = create_dataset_stats(
            num_rows=5000,
            num_columns=3,
            size_bytes=1048576,
            features={"text": "string", "label": "int64"},
        )
        assert stats.num_rows == 5000
        assert stats.num_columns == 3
        assert stats.size_bytes == 1048576
        assert stats.features["text"] == "string"

    def test_create_negative_rows_raises(self) -> None:
        """Test that negative num_rows raises ValueError."""
        with pytest.raises(ValueError, match="num_rows must be non-negative"):
            create_dataset_stats(num_rows=-1)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_formats(self) -> None:
        """Test list_formats returns sorted list."""
        formats = list_formats()
        assert "parquet" in formats
        assert "json" in formats
        assert "csv" in formats
        assert formats == sorted(formats)
        assert len(formats) == len(VALID_FORMATS)

    def test_list_splits(self) -> None:
        """Test list_splits returns sorted list."""
        splits = list_splits()
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        assert splits == sorted(splits)
        assert len(splits) == len(VALID_SPLITS)

    def test_list_streaming_modes(self) -> None:
        """Test list_streaming_modes returns sorted list."""
        modes = list_streaming_modes()
        assert "disabled" in modes
        assert "basic" in modes
        assert "shuffled" in modes
        assert modes == sorted(modes)
        assert len(modes) == len(VALID_STREAMING_MODES)


class TestGetFunctions:
    """Tests for get_* functions."""

    def test_get_format_valid(self) -> None:
        """Test get_format with valid values."""
        assert get_format("parquet") == DatasetFormat.PARQUET
        assert get_format("json") == DatasetFormat.JSON
        assert get_format("csv") == DatasetFormat.CSV
        assert get_format("arrow") == DatasetFormat.ARROW
        assert get_format("webdataset") == DatasetFormat.WEBDATASET

    def test_get_format_invalid(self) -> None:
        """Test get_format with invalid value raises."""
        with pytest.raises(ValueError, match="format must be one of"):
            get_format("invalid")

    def test_get_split_valid(self) -> None:
        """Test get_split with valid values."""
        assert get_split("train") == SplitType.TRAIN
        assert get_split("validation") == SplitType.VALIDATION
        assert get_split("test") == SplitType.TEST
        assert get_split("all") == SplitType.ALL

    def test_get_split_invalid(self) -> None:
        """Test get_split with invalid value raises."""
        with pytest.raises(ValueError, match="split must be one of"):
            get_split("invalid")

    def test_get_streaming_mode_valid(self) -> None:
        """Test get_streaming_mode with valid values."""
        assert get_streaming_mode("disabled") == StreamingMode.DISABLED
        assert get_streaming_mode("basic") == StreamingMode.BASIC
        assert get_streaming_mode("shuffled") == StreamingMode.SHUFFLED

    def test_get_streaming_mode_invalid(self) -> None:
        """Test get_streaming_mode with invalid value raises."""
        with pytest.raises(ValueError, match="streaming_mode must be one of"):
            get_streaming_mode("invalid")


class TestEstimateDownloadSize:
    """Tests for estimate_download_size function."""

    def test_basic_estimation(self) -> None:
        """Test basic download size estimation."""
        size = estimate_download_size(1000)
        # 1000 rows * 1024 bytes * 0.3 compression = 307200
        assert size == 307200

    def test_custom_row_size(self) -> None:
        """Test estimation with custom row size."""
        size = estimate_download_size(1000, avg_row_size_bytes=2048)
        # 1000 * 2048 * 0.3 = 614400
        assert size == 614400

    def test_custom_compression(self) -> None:
        """Test estimation with custom compression ratio."""
        size = estimate_download_size(1000, compression_ratio=0.5)
        # 1000 * 1024 * 0.5 = 512000
        assert size == 512000

    def test_zero_rows(self) -> None:
        """Test estimation with zero rows."""
        size = estimate_download_size(0)
        assert size == 0

    def test_negative_rows_raises(self) -> None:
        """Test that negative rows raises ValueError."""
        with pytest.raises(ValueError, match="num_rows must be non-negative"):
            estimate_download_size(-1)

    def test_negative_row_size_raises(self) -> None:
        """Test that negative row size raises ValueError."""
        with pytest.raises(ValueError, match="avg_row_size_bytes must be non-negative"):
            estimate_download_size(100, avg_row_size_bytes=-1)

    def test_invalid_compression_low_raises(self) -> None:
        """Test that compression < 0 raises ValueError."""
        with pytest.raises(ValueError, match="compression_ratio must be between"):
            estimate_download_size(100, compression_ratio=-0.1)

    def test_invalid_compression_high_raises(self) -> None:
        """Test that compression > 1 raises ValueError."""
        with pytest.raises(ValueError, match="compression_ratio must be between"):
            estimate_download_size(100, compression_ratio=1.5)

    def test_boundary_compression_values(self) -> None:
        """Test boundary compression values (0 and 1)."""
        size_zero = estimate_download_size(100, compression_ratio=0)
        assert size_zero == 0

        size_full = estimate_download_size(100, compression_ratio=1)
        assert size_full == 100 * 1024


class TestCalculateDatasetHash:
    """Tests for calculate_dataset_hash function."""

    def test_sha256_bytes(self) -> None:
        """Test SHA256 hash of bytes."""
        result = calculate_dataset_hash(b"test data")
        expected = "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"
        assert result == expected

    def test_sha256_string(self) -> None:
        """Test SHA256 hash of string."""
        result = calculate_dataset_hash("test data")
        expected = "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"
        assert result == expected

    def test_md5(self) -> None:
        """Test MD5 hash."""
        result = calculate_dataset_hash(b"test", algorithm="md5")
        assert result == "098f6bcd4621d373cade4e832627b4f6"

    def test_sha1(self) -> None:
        """Test SHA1 hash."""
        result = calculate_dataset_hash(b"test", algorithm="sha1")
        assert result == "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3"

    def test_invalid_algorithm_raises(self) -> None:
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="algorithm must be one of"):
            calculate_dataset_hash(b"data", algorithm="invalid")

    def test_empty_bytes(self) -> None:
        """Test hash of empty bytes."""
        result = calculate_dataset_hash(b"")
        # SHA256 of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected

    def test_unicode_string(self) -> None:
        """Test hash of unicode string."""
        result = calculate_dataset_hash("Hello, World!")
        assert len(result) == 64  # SHA256 hex length


class TestValidateDatasetFormat:
    """Tests for validate_dataset_format function."""

    def test_parquet_format(self) -> None:
        """Test parquet format validation."""
        assert validate_dataset_format("data.parquet", "parquet") is True
        assert validate_dataset_format("data.pq", "parquet") is True
        assert validate_dataset_format("data.json", "parquet") is False

    def test_json_format(self) -> None:
        """Test JSON format validation."""
        assert validate_dataset_format("data.json", "json") is True
        assert validate_dataset_format("data.jsonl", "json") is True
        assert validate_dataset_format("data.csv", "json") is False

    def test_csv_format(self) -> None:
        """Test CSV format validation."""
        assert validate_dataset_format("data.csv", "csv") is True
        assert validate_dataset_format("data.tsv", "csv") is True
        assert validate_dataset_format("data.json", "csv") is False

    def test_arrow_format(self) -> None:
        """Test Arrow format validation."""
        assert validate_dataset_format("data.arrow", "arrow") is True
        assert validate_dataset_format("data.feather", "arrow") is True
        assert validate_dataset_format("data.parquet", "arrow") is False

    def test_webdataset_format(self) -> None:
        """Test WebDataset format validation."""
        assert validate_dataset_format("archive.tar", "webdataset") is True
        assert validate_dataset_format("archive.tar.gz", "webdataset") is True
        assert validate_dataset_format("archive.tgz", "webdataset") is True
        assert validate_dataset_format("data.parquet", "webdataset") is False

    def test_case_insensitive(self) -> None:
        """Test case insensitive matching."""
        assert validate_dataset_format("DATA.PARQUET", "parquet") is True
        assert validate_dataset_format("Data.Json", "json") is True

    def test_invalid_format_raises(self) -> None:
        """Test that invalid expected_format raises ValueError."""
        with pytest.raises(ValueError, match="expected_format must be one of"):
            validate_dataset_format("data.txt", "invalid")


class TestCompareDatasetSchemas:
    """Tests for compare_dataset_schemas function."""

    def test_matching_schemas(self) -> None:
        """Test comparing identical schemas."""
        schema = {"text": "string", "label": "int64"}
        result = compare_dataset_schemas(schema, schema)
        assert result["text"]["status"] == "match"
        assert result["label"]["status"] == "match"

    def test_field_only_in_a(self) -> None:
        """Test field only present in first schema."""
        schema_a = {"text": "string", "label": "int64"}
        schema_b = {"text": "string"}
        result = compare_dataset_schemas(schema_a, schema_b)
        assert result["label"]["status"] == "only_in_a"
        assert result["label"]["type_a"] == "int64"
        assert result["label"]["type_b"] is None

    def test_field_only_in_b(self) -> None:
        """Test field only present in second schema."""
        schema_a = {"text": "string"}
        schema_b = {"text": "string", "score": "float32"}
        result = compare_dataset_schemas(schema_a, schema_b)
        assert result["score"]["status"] == "only_in_b"
        assert result["score"]["type_a"] is None
        assert result["score"]["type_b"] == "float32"

    def test_type_mismatch(self) -> None:
        """Test type mismatch detection."""
        schema_a = {"value": "int64"}
        schema_b = {"value": "float32"}
        result = compare_dataset_schemas(schema_a, schema_b)
        assert result["value"]["status"] == "type_mismatch"
        assert result["value"]["type_a"] == "int64"
        assert result["value"]["type_b"] == "float32"

    def test_empty_schemas(self) -> None:
        """Test comparing empty schemas."""
        result = compare_dataset_schemas({}, {})
        assert result == {}

    def test_complex_comparison(self) -> None:
        """Test complex schema comparison."""
        schema_a = {"a": "int", "b": "str", "c": "float"}
        schema_b = {"a": "int", "b": "bytes", "d": "bool"}
        result = compare_dataset_schemas(schema_a, schema_b)

        assert result["a"]["status"] == "match"
        assert result["b"]["status"] == "type_mismatch"
        assert result["c"]["status"] == "only_in_a"
        assert result["d"]["status"] == "only_in_b"


class TestFormatDatasetStats:
    """Tests for format_dataset_stats function."""

    def test_format_basic_stats(self) -> None:
        """Test formatting basic stats."""
        stats = create_dataset_stats(
            num_rows=10000,
            num_columns=5,
            size_bytes=1048576,
        )
        output = format_dataset_stats(stats)
        assert "10,000 rows" in output
        assert "5" in output
        assert "1.00 MB" in output

    def test_format_with_features(self) -> None:
        """Test formatting stats with features."""
        stats = create_dataset_stats(
            num_rows=100,
            num_columns=2,
            size_bytes=1024,
            features={"text": "string", "label": "int64"},
        )
        output = format_dataset_stats(stats)
        assert "text: string" in output
        assert "label: int64" in output

    def test_format_empty_stats(self) -> None:
        """Test formatting empty stats."""
        stats = create_dataset_stats()
        output = format_dataset_stats(stats)
        assert "0 rows" in output

    def test_format_size_bytes(self) -> None:
        """Test formatting with bytes size."""
        stats = create_dataset_stats(size_bytes=500)
        output = format_dataset_stats(stats)
        assert "500 bytes" in output

    def test_format_size_kb(self) -> None:
        """Test formatting with KB size."""
        stats = create_dataset_stats(size_bytes=2048)
        output = format_dataset_stats(stats)
        assert "KB" in output

    def test_format_size_gb(self) -> None:
        """Test formatting with GB size."""
        stats = create_dataset_stats(size_bytes=1024 * 1024 * 1024 * 2)
        output = format_dataset_stats(stats)
        assert "GB" in output


class TestGetRecommendedDatasetConfig:
    """Tests for get_recommended_dataset_config function."""

    def test_default_config(self) -> None:
        """Test default recommended config."""
        config = get_recommended_dataset_config()
        assert config.streaming == StreamingMode.DISABLED
        assert config.split == SplitType.TRAIN

    def test_large_dataset_config(self) -> None:
        """Test config for large dataset."""
        config = get_recommended_dataset_config(dataset_size="large")
        assert config.streaming == StreamingMode.SHUFFLED

    def test_medium_dataset_config(self) -> None:
        """Test config for medium dataset."""
        config = get_recommended_dataset_config(dataset_size="medium")
        assert config.streaming == StreamingMode.BASIC

    def test_evaluation_use_case(self) -> None:
        """Test config for evaluation use case."""
        config = get_recommended_dataset_config(use_case="evaluation")
        assert config.split == SplitType.VALIDATION

    def test_inference_use_case(self) -> None:
        """Test config for inference use case."""
        config = get_recommended_dataset_config(use_case="inference")
        assert config.split == SplitType.TEST

    def test_invalid_size_raises(self) -> None:
        """Test that invalid dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be one of"):
            get_recommended_dataset_config(dataset_size="invalid")

    def test_invalid_use_case_raises(self) -> None:
        """Test that invalid use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_dataset_config(use_case="invalid")

    def test_trust_remote_code_false(self) -> None:
        """Test that trust_remote_code is always False."""
        config = get_recommended_dataset_config()
        assert config.trust_remote_code is False

    def test_placeholder_name(self) -> None:
        """Test that placeholder name is set."""
        config = get_recommended_dataset_config()
        assert config.name == "__placeholder__"
