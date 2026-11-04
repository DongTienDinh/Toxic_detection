# Repository Guidelines

## Project Structure & Module Organization

This repository implements three-class Vietnamese toxic-text classification with ViSoBERT. Core pipeline modules live in `src/`, separated into data preparation, augmentation, models, training, evaluation, checkpointing, and aggregation. Experiment definitions are in `configs/`; augmentation dictionaries and allowlists belong in `resources/`. `notebooks/` contains the numbered workflow, `tests/` mirrors pipeline concerns, and `docs/` holds design and operational notes.

Generated datasets, checkpoints, manifests, and run outputs must stay in ignored directories such as `PRIVATE/`, `data/`, `runs/`, or `outputs/`.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install dependencies:

```powershell
python -m pip install -r requirements.in
python -m unittest discover -s tests -v
```

Prepare data with `python -m src.prepare_data ...`, resolve an experiment with `python -m src.config --experiment GATE --seed 42 ...`, and train one locked configuration with:

```powershell
python -m src.train --config PRIVATE/final_configs/BASE__seed42__protocol_v2.json --output-root PRIVATE/runs
```

Consult `README.md` for arguments and protocol-lock sequencing. Do not reinstall PyTorch/CUDA blindly on Kaggle; treat its runtime as authoritative.

## Coding Style & Naming Conventions

Use four-space indentation, UTF-8, type hints for public interfaces, and concise docstrings. Prefer `pathlib.Path`, explicit validation, deterministic behavior, and fail-closed errors. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Match existing experiment identifiers such as `BASE`, `GATE`, and `protocol_v2`.

## Testing Guidelines

Tests use `unittest` and may run under pytest. Name files `test_<area>.py`, classes `<Area>Tests`, and methods `test_<behavior>`. Add regression tests for hashes, splits, padding, checkpoints, metrics, augmentation QC, and test-set isolation. Use temporary directories and synthetic records; never require gated data or network access.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout. Follow Conventional Commits, for example `feat: add gated fusion ablation` or `fix: reject mismatched manifest hash`. Keep commits focused. Pull requests should state protocol impact, list changed configs, include test results, and link issues. Attach metrics when results change, but never commit secrets, private data, model weights, or notebook outputs.

## Reproducibility & Security

Pin model and dataset revisions before final runs. Preserve seeds, artifact hashes, QC approvals, and protocol locks. Test predictions are accessible only during final evaluation; do not weaken this boundary for debugging convenience.
