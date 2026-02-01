# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **PMAT-113**: Complete documentation suite for 100/100 PMAT score
  - `CONTRIBUTING.md` - Contribution guidelines with TDD workflow
  - `CODE_OF_CONDUCT.md` - Contributor Covenant v2.1
  - `SECURITY.md` - Security policy and vulnerability reporting
  - `docs/pmat-scorecard.md` - Detailed PMAT score breakdown

- **PMAT-100 through PMAT-112**: 13 advanced ML modules
  - `training/losses.py` - Focal, contrastive, label smoothing losses
  - `training/collators.py` - Padding/truncation strategies
  - `training/dynamics.py` - Loss curves, convergence detection
  - `evaluation/harness.py` - Evaluation harness, task configs
  - `evaluation/comparison.py` - Statistical tests, effect sizes
  - `inference/engines.py` - vLLM, TGI, TensorRT configs
  - `hub/telemetry.py` - Metrics, logging, export formats
  - `safety/privacy.py` - Differential privacy, PII detection
  - `deployment/cost.py` - Cloud provider cost estimation
  - `generation/constraints.py` - Grammar, regex, length constraints
  - `models/analysis.py` - Parameter counting, FLOPs, layer stats
  - `preprocessing/pipeline.py` - Stage orchestration, optimization
  - `rag/evaluation.py` - Retrieval metrics, faithfulness scoring

- **PMAT-087 through PMAT-099**: 13 advanced ML modules
  - `models/architectures.py` - Transformer configs
  - `models/layers.py` - MLP, FFN, gated layers
  - `preprocessing/vocabulary.py` - BPE training
  - `preprocessing/curation.py` - Dataset cleaning
  - `deployment/serving.py` - Serving configs
  - `deployment/conversion.py` - Model format conversion
  - `inference/memory.py` - Memory estimation
  - `inference/hardware.py` - Hardware detection
  - `hub/versioning.py` - Model versioning
  - `hub/datasets.py` - Dataset management
  - `evaluation/leaderboards.py` - Leaderboard configs
  - `training/reproducibility.py` - Seed management
  - `training/debugging.py` - Gradient visualization

- **PMAT-074 through PMAT-086**: 13 core ML modules
  - `models/attention.py` - Flash/Multi-Query/Grouped-Query Attention
  - `models/positional.py` - RoPE, ALiBi encodings
  - `models/normalization.py` - LayerNorm, RMSNorm
  - `models/activations.py` - GELU, SwiGLU
  - `training/optimizers.py` - AdamW, Lion, Sophia
  - `training/schedulers.py` - Cosine, warmup schedulers
  - `training/gradient.py` - Gradient accumulation/clipping
  - `inference/decoding.py` - Beam search, nucleus sampling
  - `generation/prompting.py` - Prompt templates, CoT
  - `generation/tools.py` - Function calling, tool use
  - `evaluation/benchmarks.py` - MMLU, HellaSwag
  - `evaluation/metrics.py` - BLEU, ROUGE, BERTScore
  - `preprocessing/sampling.py` - Stratified, importance sampling

- **PMAT-061 through PMAT-073**: 13 advanced modules
  - Quantization, pruning, NAS, hyperopt, active learning
  - Meta-learning, multi-task, hybrid search, calibration
  - Knowledge editing, model cards, embeddings, data filtering

- **PMAT-048 through PMAT-060**: 13 foundation modules
  - Adapters, merging, checkpointing, mixed precision, parallelism
  - Augmentation, tokenization, synthetic data, vectorstore, chunking
  - Caching, context extension, profiling

- **PMAT-001 through PMAT-047**: Core infrastructure and initial modules
  - Hub search, inference pipelines, preprocessing, training, evaluation
  - Deployment optimization, audio, multimodal, safety, RAG, generation
  - Continuous batching, speculative decoding, KV cache, MoE
  - Data quality, bias detection, robustness testing, experiment tracking

### Fixed

- Type checker errors across 8 source modules
- Security scan findings (MD5, pickle, hardcoded paths)
- Hypothesis filter_too_much health check failures
- Lint violations (E501, RUF043, SIM102, F401, F811)

## [0.1.0] - 2026-01-30

### Added

- Initial project structure with `uv` package management
- Core `hf_gtc` package with hub, inference, preprocessing modules
- Quality gates: lint, format, typecheck, security, coverage
- Pre-commit hooks with PMAT integration
- CI/CD pipeline with GitHub Actions

[Unreleased]: https://github.com/noahgift/hf-ground-truth-corpus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/noahgift/hf-ground-truth-corpus/releases/tag/v0.1.0
