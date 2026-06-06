# KAN-D-IV-LATE

Research code and manuscript materials for the paper  
**“Rethinking Distributional IVs: KAN-Powered D-IV-LATE & Model Choice.”**

## Current Status

This repository has been rationalised into one reviewer-usable path.

- The importable package is [`src/kan_d_iv_late/`](./src/kan_d_iv_late).
- The active research code path is [`kan-d-iv-late/`](./kan-d-iv-late).
- Historical duplicate trees live under `legacy/` and are not the canonical execution path.
- Publication planning and execution state live under [`.planning/`](./.planning).
- Corrected paper-facing evidence is stored under
  `kan-d-iv-late/results/corrected_2026_06_06/`.

## Active Layout

```text
.
├── src/kan_d_iv_late/      # Importable Python package
├── kan-d-iv-late/          # CLI wrappers, data, and results path
├── .planning/              # GSD-style publication planning workspace
├── paper/                  # Paper draft and compiled PDF
└── legacy/                 # Archived duplicate project trees (after rationalisation)
```

## Why The Repo Is Being Reorganised

The repo historically accumulated multiple overlapping project bundles, including:

- a baseline D-IV-LATE tree
- a KAN-focused tree
- a `kan-d-iv-late-main/` bundle containing further duplicates and notebook work

That duplication made it hard to know which scripts, data, and outputs are authoritative. The cleanup has established one canonical implementation path so publication-focused experiments can proceed from a single codebase.

## Canonical Setup

For local development with the sibling `efficient-kan` checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

For a non-local install, the project metadata and legacy requirements file pin
`efficient-kan` to the commit used for the corrected Phase 5 evidence:
`c07843613e9e5c09523e09c000038128bc41cde4`.

## Canonical Commands

Empirical pipeline:

```bash
python kan-d-iv-late/run_empirical.py
```

Simulation pipeline:

```bash
python kan-d-iv-late/run_simulation.py
```

Quick simulation smoke run:

```bash
python kan-d-iv-late/run_simulation.py --quick
```

Versioned simulation artifacts are written under `kan-d-iv-late/results/simulation_runs/`
by default. The corrected paper-facing Phase 5 rerun is under
`kan-d-iv-late/results/corrected_2026_06_06/`.

Current test gate:

```bash
pytest -q tests
```

## Current Caveat

The importable package is now the active implementation surface. The
`kan-d-iv-late/` directory remains the canonical data/results and CLI-wrapper
location. The main numerical evidence has been regenerated under the corrected
score path; outstanding publication work is now robustness/calibration and
final submission polish, not another score correction.

Do not rerun the full evidence suite casually. The corrected Phase 5 run took
approximately 12 hours, and future evidence refreshes should be deliberate.

## Local Validation

This project intentionally does not use CI/CD. Run the local validation gate
before committing implementation changes:

```bash
scripts/validate.sh
```

The gate compiles Python sources, runs the test suite, and checks whitespace. It
does not run expensive simulation or empirical evidence jobs.
