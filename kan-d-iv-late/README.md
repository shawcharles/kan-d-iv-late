# D-IV-LATE CLI, Data, and Artifact Directory

This directory contains the command-line wrappers, input data, and default
artifact locations for the `kan_d_iv_late` Python package.

The active implementation now lives in the importable package at
`../src/kan_d_iv_late/`. Keep estimator and nuisance-learning changes there. The
scripts in this directory are thin execution entrypoints retained for stable
research commands and artifact paths.

The files under `code/` are compatibility shims for older dynamic-import users.
They should not receive new implementation logic.

## Commands

Run from the repository root after installing with `requirements-local.txt`:

```bash
python kan-d-iv-late/run_simulation.py --quick
python kan-d-iv-late/run_simulation_matrix.py --profile smoke
python kan-d-iv-late/run_empirical.py --profile core
python kan-d-iv-late/run_inference_validation.py --profile smoke
```

## Dependency Policy

For local development, use the sibling checkout of `efficient-kan`:

```bash
python -m pip install -r requirements-local.txt
```

For non-local installs, `kan-d-iv-late/requirements.txt` pins the Git
dependency to the `efficient-kan` commit used by the corrected evidence runs.

## Artifact Policy

New generated outputs under `results/simulation_runs/`,
`results/empirical_runs/`, `results/inference_runs/`, and
`results/corrected_*` are ignored by default. Record final evidence through
manifests and paper tables rather than committing ad hoc smoke-run artifacts.
