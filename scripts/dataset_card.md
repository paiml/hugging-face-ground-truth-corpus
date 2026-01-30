---
license: apache-2.0
task_categories:
  - text-generation
  - text2text-generation
language:
  - en
size_categories:
  - 1K<n<10K
tags:
  - huggingface
  - transformers
  - ground-truth
  - recipes
  - peft
  - lora
  - qlora
  - depyler
  - code
  - python
---

# HuggingFace Ground Truth Corpus (HF-GTC)

Curated Python recipes for HuggingFace ML patterns with 98.46% test coverage.

## Dataset Description

HF-GTC is a collection of high-quality Python code implementing common HuggingFace patterns:

- **Hub Operations**: Cards, repositories, Spaces API
- **Preprocessing**: Tokenization, streaming, augmentation
- **Training**: Fine-tuning, LoRA, QLoRA, callbacks
- **Inference**: Pipeline operations, batch processing
- **Evaluation**: Metrics, benchmarks, leaderboards
- **Deployment**: Quantization, GGUF export, serving

Each module includes:
- Comprehensive docstrings with examples
- Type annotations for static analysis
- Doctests for inline verification
- Property-based test coverage

## Statistics

| Metric | Value |
|--------|-------|
| Modules | 22 |
| Functions | ~180 |
| Tests | 1258 |
| Coverage | 98.46% |
| Statements | 2726 |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `module` | `string` | Module path (e.g., `training/lora.py`) |
| `category` | `string` | Category name (hub, inference, etc.) |
| `content` | `string` | Full module source code |
| `functions` | `list<struct>` | Extracted function metadata |
| `doctests` | `list<struct>` | Extracted doctest examples |
| `coverage` | `float32` | Module test coverage percentage |
| `test_count` | `int32` | Number of tests for module |

### Functions Schema

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Function name |
| `signature` | `string` | Type signature |
| `docstring` | `string` | Full docstring with examples |
| `line_number` | `int32` | Source line number |

### Doctests Schema

| Field | Type | Description |
|-------|------|-------------|
| `source` | `string` | Doctest input code |
| `expected` | `string` | Expected output |
| `line_number` | `int32` | Source line number |

## Usage

```python
from datasets import load_dataset

# Load the corpus
corpus = load_dataset("paiml/hf-ground-truth-corpus")

# Access a training module
training_modules = corpus.filter(lambda x: x["category"] == "training")

# Extract LoRA examples
for module in training_modules:
    if "lora" in module["module"]:
        print(f"Module: {module['module']}")
        print(f"Functions: {len(module['functions'])}")
        print(f"Coverage: {module['coverage']:.1f}%")
```

## Use Cases

1. **Code Generation Training**: Fine-tune models on high-quality HuggingFace patterns
2. **Documentation Generation**: Train models to generate docstrings from code
3. **Test Generation**: Learn to generate doctests from function signatures
4. **Code Completion**: Pattern-aware completions for ML workflows
5. **Depyler Transpilation**: Python-to-Rust code transformation corpus

## Quality Guarantees

This corpus is produced with:

- **TDD Methodology**: All code is test-first with 95%+ coverage
- **Popperian Falsification**: Red team audited for specification claims
- **Static Typing**: Full type annotations for `ty` compatibility
- **Unicode Safety**: NFC normalization for consistent tokenization

## Citation

```bibtex
@dataset{hf_gtc_2026,
  title={HuggingFace Ground Truth Corpus},
  author={PAIML},
  year={2026},
  publisher={HuggingFace Hub},
  url={https://huggingface.co/datasets/paiml/hf-ground-truth-corpus}
}
```

## License

Apache 2.0 - See LICENSE file for details.
