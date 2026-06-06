from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kan_d_iv_late import simulation

PROJECT_DIR = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Run the canonical simulation benchmark pipeline.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results" / "simulation_runs")
    parser.add_argument("--n-simulations", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--design", default="baseline")
    parser.add_argument("--instrument-strength", default="medium")
    parser.add_argument("--y-points", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--kan-steps", type=int, default=25)
    parser.add_argument("--kan-hidden-dim", type=int, default=16)
    parser.add_argument("--kan-grid-size", type=int, default=4)
    parser.add_argument("--kan-spline-order", type=int, default=3)
    parser.add_argument("--kan-lr", type=float, default=1e-3)
    parser.add_argument("--kan-weight-decay", type=float, default=1e-4)
    parser.add_argument("--kan-reg-strength", type=float, default=1e-4)
    parser.add_argument("--kan-min-class-count", type=int, default=5)
    parser.add_argument("--probability-epsilon", type=float, default=1e-6)
    parser.add_argument("--truth-sample-size", type=int, default=50000)
    parser.add_argument("--truth-seed", type=int, default=1729)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--tag")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a much smaller benchmark for smoke testing and iteration.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.quick:
        args.n_simulations = min(args.n_simulations, 2)
        args.n_samples = min(args.n_samples, 200)
        args.y_points = min(args.y_points, 5)
        args.kan_steps = min(args.kan_steps, 5)
        args.truth_sample_size = min(args.truth_sample_size, 20000)
        if args.tag is None:
            args.tag = "quick"

    simulation.main(
        results_dir=args.results_dir,
        n_simulations=args.n_simulations,
        n_samples=args.n_samples,
        design_name=args.design,
        instrument_strength=args.instrument_strength,
        y_points=args.y_points,
        k_folds=args.folds,
        kan_steps=args.kan_steps,
        kan_hidden_dim=args.kan_hidden_dim,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        kan_lr=args.kan_lr,
        kan_weight_decay=args.kan_weight_decay,
        kan_reg_strength=args.kan_reg_strength,
        kan_min_class_count=args.kan_min_class_count,
        probability_epsilon=args.probability_epsilon,
        truth_sample_size=args.truth_sample_size,
        truth_seed=args.truth_seed,
        seed_base=args.seed,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
