# KAN-D-IV-LATE

Research code and supplementary materials for the paper  
**“Rethinking Distributional IVs: KAN-Powered D-IV-LATE & Model Choice”**  
([arXiv:2506.12765](https://arxiv.org/abs/2506.12765)).

## Project Overview
This repository implements two double/debiased-machine-learning estimators for the Distributional Instrumental-Variable Local Average Treatment Effect (D-IV-LATE):

1. **RF-D-IV-LATE** – baseline estimator that uses Random Forests for all nuisance functions.
2. **KAN-D-IV-LATE** – novel estimator that replaces the Random Forests with Kolmogorov-Arnold Networks (KANs).

The codebase reproduces the Monte-Carlo simulation study and the empirical application to the 401(k) pension data reported in the paper.

## Directory Structure
```text
kan-d-iv-late-main/
├── kan-d-iv-late/
│   ├── code/
│   │   ├── kan-d-iv-late_simulation.py          # Monte-Carlo study
│   │   └── kan-d-iv-late_empirical_application.py
│   └── requirements.txt
├── KAN-D-IV-LATE/data/pension.csv               # 401(k) data
├── paper/main.tex                               # TeX source of the article
└── README.md
```

## Getting Started

### 1. Clone and install dependencies
```bash
git clone https://github.com/<username>/kan-d-iv-late.git
cd kan-d-iv-late-main
python -m venv .venv
source .venv/bin/activate
pip install -r kan-d-iv-late/requirements.txt
# If you want the bleeding-edge KAN implementation:
pip install efficient-kan   # optional; used by the simulation script
```

### 2. Monte-Carlo Simulation
```bash
python kan-d-iv-late/code/kan-d-iv-late_simulation.py
```
Outputs → `simulation_results.csv` (and a ZIP archive if running in Google Colab).

### 3. Empirical Application
Ensure the pension dataset is present at `KAN-D-IV-LATE/data/pension.csv` (already supplied).
```bash
python kan-d-iv-late/code/kan-d-iv-late_empirical_application.py
```
Outputs →
* `KAN-D-IV-LATE/results/empirical_kan-d-iv-late_results.csv` – pointwise D-IV-LATE estimates  
* `KAN-D-IV-LATE/results/empirical_kan-d-iv-late_plot.png` – publication-ready plot of the D-IV-LATE curve

## Reproducing Paper Figures
Running the two scripts above will save all the tables and figures used in the manuscript to `KAN-D-IV-LATE/results/`.  
You can re-compile the paper with:
```bash
cd paper
latexmk -pdf main.tex
```

## Citation
If you use this code, please cite
```bibtex
@article{Shaw2025KANDIVLATE,
  title   = {Rethinking Distributional IV: KAN-Powered D-IV-LATE & Model Choice},
  author  = {Charles Shaw},
  journal = {arXiv preprint arXiv:2506.12765},
  year    = {2025}
}
```

## License
Apache 2.0 — see [`LICENSE`](LICENSE) for details.

## Contact
Questions or suggestions? Open an issue or email <charles@fixedpoint.io>.

