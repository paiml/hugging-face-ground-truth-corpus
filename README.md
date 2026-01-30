# HuggingFace Ground Truth Corpus

<div align="center">

<img src="docs/assets/hero.svg" alt="HuggingFace Ground Truth Corpus" width="800"/>

[![CI](https://github.com/paiml/hugging-face-ground-truth-corpus/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/hugging-face-ground-truth-corpus/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/paiml/hugging-face-ground-truth-corpus)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready ML recipes with 95%+ test coverage**

</div>

## Overview

A curated collection of **production-ready Python recipes** for HuggingFace ML workflows, built with the highest engineering standards:

- **95%+ Test Coverage** enforced via pytest
- **Property-Based Testing** with Hypothesis
- **Zero Linting Violations** via ruff
- **Toyota Production System** quality methodology
- **Popperian Falsification** test philosophy

## Installation

```bash
# Clone the repository
git clone https://github.com/paiml/hugging-face-ground-truth-corpus.git
cd hugging-face-ground-truth-corpus

# Install with uv (required)
uv sync --extra dev
```

## Quick Start

```python
from hf_gtc.hub.search import search_models, search_datasets
from hf_gtc.inference.pipelines import create_pipeline
from hf_gtc.preprocessing.tokenization import preprocess_text

# Search for models
models = search_models(task="text-classification", limit=5)
for model in models:
    print(f"{model.model_id}: {model.downloads} downloads")

# Create inference pipeline
pipe = create_pipeline("sentiment-analysis")
result = pipe("I love this library!")

# Preprocess text
clean_text = preprocess_text("  HELLO   WORLD  ")
# Returns: "hello world"
```

## Modules

| Module | Description |
|--------|-------------|
| `hf_gtc.hub` | HuggingFace Hub search and discovery |
| `hf_gtc.inference` | Device management and pipelines |
| `hf_gtc.preprocessing` | Text preprocessing and tokenization |
| `hf_gtc.training` | Fine-tuning utilities |
| `hf_gtc.evaluation` | Metrics and evaluation |
| `hf_gtc.deployment` | Model optimization |

## Development

```bash
# Run tests with coverage
uv run pytest --cov-fail-under=95

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Security scan
uv run bandit -r src/ -ll
```

## Quality Gates (Jidoka)

All PRs must pass:

1. **Gate 1** - Lint (ruff check)
2. **Gate 2** - Format (ruff format --check)
3. **Gate 3** - Security (bandit)
4. **Gate 4** - Tests + Coverage (95% minimum)

## Architecture

```
src/hf_gtc/
├── hub/           # HuggingFace Hub integration
│   └── search.py  # Model/dataset search
├── inference/     # Inference utilities
│   ├── device.py  # GPU/CPU device management
│   └── pipelines.py
├── preprocessing/ # Data preprocessing
│   └── tokenization.py
├── training/      # Training recipes
│   └── fine_tuning.py
├── evaluation/    # Metrics
└── deployment/    # Optimization
```

## Querying from Batuta / Aprender

This corpus serves as **ground truth** for the Sovereign AI Stack. Query recipes and get Rust equivalents:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  YOUR QUESTION   │────▶│  BATUTA ORACLE   │────▶│  RUST SOLUTION   │
│  "tokenize text" │     │  (RAG search)    │     │  via candle      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### From Batuta (Oracle Queries)

```bash
# Natural language query
batuta oracle "How do I tokenize text for BERT?"
# Returns: hf_gtc/preprocessing/tokenization.py + candle equivalent

# Query with Rust cross-reference
batuta oracle --rust-source candle "attention mechanism"

# Query by tag
batuta oracle --tag training --tag memory-efficient
```

### From Aprender (Rust ML)

```rust
// Python recipe in hf_gtc:
//   from hf_gtc.preprocessing import preprocess_text
//   result = preprocess_text("  HELLO  ")  # "hello"

// Equivalent Rust (via Depyler transpilation):
let result = preprocess_text("  HELLO  ");  // "hello"
```

### Depyler Transpilation

Qualified recipes (MQS ≥ 85) can be transpiled to Rust:

```bash
# Transpile Python recipes to Rust
depyler transpile src/hf_gtc/ --output rust_output/ --verify

# Verify semantic equivalence against candle
depyler verify --python src/hf_gtc/preprocessing/ --rust candle-core/
```

See [docs/specifications/hf-ground-truth-corpus.md](docs/specifications/hf-ground-truth-corpus.md) for full integration details.

## Rust Ground Truth

This project cross-references HuggingFace's Rust implementations for validation:

- **[candle](https://github.com/huggingface/candle)** - Tensor operations
- **[safetensors](https://github.com/huggingface/safetensors)** - Safe serialization

## References

- Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*
- Popper, K. (1959). *The Logic of Scientific Discovery*
- Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with the PAIML team</sub>
</div>
