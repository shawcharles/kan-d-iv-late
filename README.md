# KAN-D-IV-LATE

Research code and manuscript materials for the paper  
**“Rethinking Distributional IVs: KAN-Powered D-IV-LATE & Model Choice.”**

## Current Status

This repository has been rationalised into one reviewer-usable path.

- The active research code path is [`kan-d-iv-late/`](./kan-d-iv-late).
- Historical duplicate trees live under `legacy/` and are not the canonical execution path.
- Publication planning and execution state live under [`.planning/`](./.planning).

## Active Layout

```text
.
├── kan-d-iv-late/          # Active code, data, and results path
├── .planning/              # GSD-style publication planning workspace
├── main.tex                # Paper draft
├── preamble.tex
└── legacy/                 # Archived duplicate project trees (after rationalisation)
```

## Why The Repo Is Being Reorganised

The repo historically accumulated multiple overlapping project bundles, including:

- a baseline D-IV-LATE tree
- a KAN-focused tree
- a `kan-d-iv-late-main/` bundle containing further duplicates and notebook work

That duplication made it hard to know which scripts, data, and outputs are authoritative. The cleanup has established one canonical implementation path so publication-focused experiments can proceed from a single codebase.

## Canonical Setup

Use one environment path for the active tree:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kan-d-iv-late/requirements.txt
```

The active KAN backend is `efficient-kan`, installed via `kan-d-iv-late/requirements.txt`.

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

Versioned simulation artifacts are written under `kan-d-iv-late/results/simulation_runs/`.

Current test gate:

```bash
pytest -q
```

## Current Caveat

The active code path is now canonical through Phase 2 execution. The remaining publication risks are the missing experiment matrix, empirical robustness work, inference validation, and manuscript revision.
