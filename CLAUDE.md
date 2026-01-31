# HF Ground Truth Corpus - Development Guidelines

## Project Overview

This is the **HuggingFace Ground Truth Corpus** (HF-GTC), a curated collection of Python recipes implementing HuggingFace ML patterns with the highest standards of engineering excellence.

## Critical Rules

### Package Management
- **ONLY use `uv`** - No pip, conda, or poetry
- All commands use `uv run` prefix
- Dependencies managed in `pyproject.toml`

### Quality Standards
- **95% minimum test coverage** - Enforced via pytest
- **Zero ruff violations** - `make lint` must pass
- **Zero ty type errors** - `make typecheck` must pass
- **100% docstring coverage** for public APIs
- **Property-based testing** via Hypothesis for all pure functions

### TDD Workflow
1. Write failing test first (RED)
2. Implement minimum code to pass (GREEN)
3. Refactor while maintaining green (REFACTOR)

### Doctest Requirements
Every public function MUST have:
1. **Happy path** doctest example
2. **Edge case** doctest (empty input, boundaries)
3. **Error case** doctest with `+IGNORE_EXCEPTION_DETAIL`

### Commit Format
```
feat|fix|docs|refactor|test: message (Refs PMAT-XXXX)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Commands

```bash
make setup          # Install dependencies + pre-commit
make lint           # Run ruff linter
make format         # Auto-fix formatting
make typecheck      # Run ty type checker
make test           # Full test suite with coverage
make test-fast      # Quick unit tests, no coverage
make coverage       # Generate HTML coverage report
make check          # Full quality gates (lint + typecheck + test + security)
make security       # Run bandit security scan
```

## Module Structure

Each recipe module follows this pattern:

```python
"""Module docstring with overview.

Examples:
    >>> from hf_gtc.[category] import [function]
    >>> result = [function](...)
    >>> assert result is not None
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

def function_name(param: str) -> ReturnType:
    """One-line summary.

    Args:
        param: Description.

    Returns:
        Description.

    Raises:
        ValueError: When param is invalid.

    Examples:
        >>> function_name("test")
        'expected'
    """
    ...
```

## Rust Ground Truth

When implementing recipes with candle/safetensors equivalents:
1. Cross-reference with `../candle` for tensor operations
2. Cross-reference with `../safetensors` for serialization
3. Ensure numeric results match within epsilon (1e-6)


## Stack Documentation Search

Query this corpus and the entire Sovereign AI Stack using batuta's RAG Oracle:

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search for Python ML patterns with Rust equivalents
batuta oracle --rag "tokenization for BERT"
batuta oracle --rag "sentiment analysis pipeline"
batuta oracle --rag "HuggingFace model loading"

# Check index status
batuta oracle --rag-stats
```

This corpus is indexed alongside Rust stack components, enabling cross-language pattern discovery.
