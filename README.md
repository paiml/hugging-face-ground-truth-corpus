# HF Ground Truth Corpus

A curated collection of high-quality Python recipes implementing HuggingFace ML patterns with the highest standards of engineering excellence.

[![CI](https://github.com/your-org/hf-ground-truth-corpus/workflows/CI/badge.svg)](https://github.com/your-org/hf-ground-truth-corpus/actions)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/your-org/hf-ground-truth-corpus)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Overview

HF-GTC provides a **single source of truth** for HuggingFace patterns that:

1. **Guarantees quality** via PMAT compliance (95% coverage, zero lint violations)
2. **Corroborates correctness** via doctest-driven semantic equivalence checks
3. **Enables transpilation** to Rust via Depyler qualification pipeline
4. **Powers discovery** via Batuta oracle's RAG-based retrieval

## Installation

```bash
# Using uv (required)
uv sync --all-extras

# For development
uv sync --all-extras
uv run pre-commit install
```

## Quick Start

```python
from hf_gtc.hub import search_models
from hf_gtc.inference import get_device, create_pipeline
from hf_gtc.preprocessing import preprocess_text

# Search for models
models = search_models(task="text-classification", limit=5)

# Check available device
device = get_device()  # Returns "cuda", "mps", or "cpu"

# Create a pipeline
classifier = create_pipeline("sentiment-analysis")
result = classifier("I love this library!")

# Preprocess text
clean_text = preprocess_text("  HELLO  WORLD  ")  # "hello world"
```

## Modules

| Module | Description |
|--------|-------------|
| `hf_gtc.hub` | Model/dataset search and Hub API utilities |
| `hf_gtc.inference` | Pipeline creation and device management |
| `hf_gtc.preprocessing` | Text preprocessing and tokenization |
| `hf_gtc.training` | Trainer API and fine-tuning utilities |
| `hf_gtc.evaluation` | Metrics and benchmark utilities |
| `hf_gtc.deployment` | Quantization and serving utilities |

## Quality Standards

- **95% minimum test coverage** (enforced)
- **Zero ruff violations** (enforced)
- **100% docstring coverage** for public APIs
- **Property-based testing** via Hypothesis

## Commands

```bash
make setup          # Install dependencies
make lint           # Run linter
make test           # Full test suite
make check          # All quality gates
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
