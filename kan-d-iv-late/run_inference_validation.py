from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CODE_DIR = PROJECT_DIR / "code"
SIMULATION_SCRIPT = CODE_DIR / "kan-d-iv-late_simulation.py"
INFERENCE_SCRIPT = CODE_DIR / "dlate_inference.py"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "inference_runs"
DEFAULT_TRUTH_SEED = 1729
DEFAULT_SEED_BASE = 20260420
POINTWISE_COLUMNS = [
    "profile_name",
    "scenario_label",
    "design_name",
    "instrument_strength",
    "n_samples",
    "model",
    "replication",
    "y_value",
    "true_dlate",
    "asymptotic_point_estimate",
    "asymptotic_ci_lower",
    "asymptotic_ci_upper",
    "asymptotic_covers",
    "asymptotic_width",
    "bootstrap_point_estimate",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "bootstrap_covers",
    "bootstrap_width",
    "mean_psi_beta",
    "near_zero_denominator",
]
SUMMARY_COLUMNS = [
    "profile_name",
    "scenario_label",
    "model",
    "replications",
    "asymptotic_coverage_mean",
    "bootstrap_coverage_mean",
    "asymptotic_width_mean",
    "bootstrap_width_mean",
    "mean_psi_beta",
    "near_zero_denominator_share",
    "asymptotic_coverage_gap",
    "bootstrap_coverage_gap",
    "recommended_method",
]


def load_module(path, module_name):
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_inference_profile(profile_name):
    if profile_name == "smoke":
        return {
            "scenarios": [
                {"design_name": "smooth_low", "instrument_strength": "weak", "n_samples": 200},
            ],
            "n_replications": 1,
            "n_bootstrap": 5,
            "y_points": 4,
            "truth_sample_size": 5000,
            "k_folds": 2,
            "kan_steps": 5,
        }
    if profile_name == "pilot":
        return {
            "scenarios": [
                {"design_name": "smooth_low", "instrument_strength": "weak", "n_samples": 500},
                {"design_name": "complex_local", "instrument_strength": "weak", "n_samples": 500},
            ],
            "n_replications": 5,
            "n_bootstrap": 10,
            "y_points": 6,
            "truth_sample_size": 20000,
            "k_folds": 3,
            "kan_steps": 25,
        }
    if profile_name == "full":
        return {
            "scenarios": [
                {"design_name": "smooth_low", "instrument_strength": "weak", "n_samples": 500},
                {"design_name": "baseline", "instrument_strength": "medium", "n_samples": 1000},
                {"design_name": "complex_local", "instrument_strength": "weak", "n_samples": 500},
            ],
            "n_replications": 8,
            "n_bootstrap": 15,
            "y_points": 8,
            "truth_sample_size": 30000,
            "k_folds": 3,
            "kan_steps": 25,
        }
    raise ValueError(f"Unsupported inference profile: {profile_name}")


def scenario_label(scenario):
    return (
        f"{scenario['design_name']}__"
        f"{scenario['instrument_strength']}__"
        f"n{scenario['n_samples']}"
    )


def select_scenarios(scenarios, *, scenario_offset=0, scenario_count=None):
    selected = scenarios[scenario_offset:]
    if scenario_count is not None:
        selected = selected[:scenario_count]
    return selected


def summarize_inference_outputs(pointwise_df):
    if pointwise_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        pointwise_df.groupby(["profile_name", "scenario_label", "model"], as_index=False)
        .agg(
            replications=("replication", "nunique"),
            asymptotic_coverage_mean=("asymptotic_covers", "mean"),
            bootstrap_coverage_mean=("bootstrap_covers", "mean"),
            asymptotic_width_mean=("asymptotic_width", "mean"),
            bootstrap_width_mean=("bootstrap_width", "mean"),
            mean_psi_beta=("mean_psi_beta", "mean"),
            near_zero_denominator_share=("near_zero_denominator", "mean"),
        )
    )
    summary["asymptotic_coverage_gap"] = np.abs(summary["asymptotic_coverage_mean"] - 0.95)
    summary["bootstrap_coverage_gap"] = np.abs(summary["bootstrap_coverage_mean"] - 0.95)
    summary["recommended_method"] = np.where(
        summary["bootstrap_coverage_gap"] < summary["asymptotic_coverage_gap"],
        "bootstrap",
        "asymptotic",
    )
    tie_mask = summary["bootstrap_coverage_gap"] == summary["asymptotic_coverage_gap"]
    summary.loc[tie_mask, "recommended_method"] = np.where(
        summary.loc[tie_mask, "bootstrap_width_mean"] <= summary.loc[tie_mask, "asymptotic_width_mean"],
        "bootstrap",
        "asymptotic",
    )
    return summary


def _load_existing_pointwise_rows(pointwise_path):
    if not pointwise_path.exists():
        return []
    pointwise_df = pd.read_csv(pointwise_path)
    if pointwise_df.empty:
        return []
    return pointwise_df.to_dict(orient="records")


def _completed_replication_keys(pointwise_rows):
    return {
        (row["scenario_label"], row["model"], int(row["replication"]))
        for row in pointwise_rows
    }


def _write_outputs(
    *,
    profile_name,
    alpha,
    profile,
    results_dir,
    pointwise_rows,
    selected_scenarios,
    scenario_offset,
    scenario_count,
):
    pointwise_path = results_dir / "inference_pointwise_results.csv"
    summary_path = results_dir / "inference_summary.csv"
    manifest_path = results_dir / f"inference_manifest_{profile_name}.json"

    pointwise_df = pd.DataFrame(pointwise_rows, columns=POINTWISE_COLUMNS)
    summary_df = summarize_inference_outputs(pointwise_df)

    pointwise_df.to_csv(pointwise_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    completed_keys = _completed_replication_keys(pointwise_rows)
    manifest = {
        "profile": profile_name,
        "alpha": alpha,
        "scenarios": profile["scenarios"],
        "scenario_count": len(profile["scenarios"]),
        "n_replications": profile["n_replications"],
        "n_bootstrap": profile["n_bootstrap"],
        "completed_replication_count": len(completed_keys),
        "last_selection": {
            "scenario_offset": scenario_offset,
            "scenario_count": scenario_count,
            "selected_scenario_labels": [scenario_label(item) for item in selected_scenarios],
        },
        "output_files": {
            "pointwise_results": str(pointwise_path),
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "pointwise_path": pointwise_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
    }


def run_profile(
    profile_name="pilot",
    *,
    results_dir=DEFAULT_RESULTS_DIR,
    alpha=0.05,
    simulation_module=None,
    inference_module=None,
    resume=False,
    scenario_offset=0,
    scenario_count=None,
):
    simulation = simulation_module or load_module(SIMULATION_SCRIPT, "kan_d_iv_late_simulation")
    inference = inference_module or load_module(INFERENCE_SCRIPT, "dlate_inference")

    profile = build_inference_profile(profile_name)
    results_dir = Path(results_dir) / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_scenarios = select_scenarios(
        profile["scenarios"],
        scenario_offset=scenario_offset,
        scenario_count=scenario_count,
    )

    pointwise_path = results_dir / "inference_pointwise_results.csv"
    pointwise_rows = _load_existing_pointwise_rows(pointwise_path) if resume else []
    completed_keys = _completed_replication_keys(pointwise_rows)

    for scenario_index, scenario in enumerate(profile["scenarios"]):
        if scenario not in selected_scenarios:
            continue

        scenario_seed_base = DEFAULT_SEED_BASE + 1000 * scenario_index
        label = scenario_label(scenario)
        truth_bundle = simulation.build_truth_bundle(
            design_name=scenario["design_name"],
            instrument_strength=scenario["instrument_strength"],
            y_points=profile["y_points"],
            truth_sample_size=profile["truth_sample_size"],
            truth_seed=DEFAULT_TRUTH_SEED,
        )
        y_grid = truth_bundle["truth_df"]["y"].to_numpy()
        true_dlate = truth_bundle["truth_df"]["true_dlate"].to_numpy()

        for model_type in ("kan", "rf"):
            for replication in range(profile["n_replications"]):
                completed_key = (label, model_type, replication)
                if completed_key in completed_keys:
                    print(f"Resuming completed inference replication: {label} / {model_type} / rep {replication}")
                    continue

                print(f"Running inference validation: {label} / {model_type} / rep {replication}")
                data, _ = simulation.generate_dlate_data(
                    n_samples=scenario["n_samples"],
                    seed=scenario_seed_base + replication,
                    design_name=scenario["design_name"],
                    instrument_strength=scenario["instrument_strength"],
                )
                nuisance_results = simulation.estimate_nuisance_functions(
                    data,
                    y_grid,
                    model_type=model_type,
                    k_folds=profile["k_folds"],
                    kan_steps=profile["kan_steps"],
                )
                asymptotic = inference.dlate_asymptotic_inference(
                    data,
                    nuisance_results,
                    y_grid,
                    alpha=alpha,
                    epsilon=simulation.PROBABILITY_EPSILON,
                )

                def nuisance_estimator(bootstrap_data, bootstrap_y_grid):
                    return simulation.estimate_nuisance_functions(
                        bootstrap_data,
                        bootstrap_y_grid,
                        model_type=model_type,
                        k_folds=profile["k_folds"],
                        kan_steps=profile["kan_steps"],
                    )

                bootstrap = inference.bootstrap_dlate_inference(
                    data,
                    y_grid,
                    nuisance_estimator,
                    alpha=alpha,
                    n_bootstrap=profile["n_bootstrap"],
                    random_state=scenario_seed_base + 10000 + replication,
                    epsilon=simulation.PROBABILITY_EPSILON,
                )
                asymptotic_coverage = inference.summarize_interval_coverage(
                    asymptotic["point_estimates"],
                    asymptotic["ci_lower"],
                    asymptotic["ci_upper"],
                    true_dlate,
                )
                bootstrap_coverage = inference.summarize_interval_coverage(
                    bootstrap["point_estimates"],
                    bootstrap["ci_lower"],
                    bootstrap["ci_upper"],
                    true_dlate,
                )

                for y_idx, y_value in enumerate(y_grid):
                    pointwise_rows.append(
                        {
                            "profile_name": profile_name,
                            "scenario_label": label,
                            "design_name": scenario["design_name"],
                            "instrument_strength": scenario["instrument_strength"],
                            "n_samples": scenario["n_samples"],
                            "model": model_type,
                            "replication": replication,
                            "y_value": y_value,
                            "true_dlate": true_dlate[y_idx],
                            "asymptotic_point_estimate": asymptotic["point_estimates"][y_idx],
                            "asymptotic_ci_lower": asymptotic["ci_lower"][y_idx],
                            "asymptotic_ci_upper": asymptotic["ci_upper"][y_idx],
                            "asymptotic_covers": int(asymptotic_coverage["covers"][y_idx]),
                            "asymptotic_width": asymptotic_coverage["widths"][y_idx],
                            "bootstrap_point_estimate": bootstrap["point_estimates"][y_idx],
                            "bootstrap_ci_lower": bootstrap["ci_lower"][y_idx],
                            "bootstrap_ci_upper": bootstrap["ci_upper"][y_idx],
                            "bootstrap_covers": int(bootstrap_coverage["covers"][y_idx]),
                            "bootstrap_width": bootstrap_coverage["widths"][y_idx],
                            "mean_psi_beta": asymptotic["mean_psi_beta"],
                            "near_zero_denominator": int(asymptotic["near_zero_denominator"]),
                        }
                    )

                completed_keys.add(completed_key)
                _write_outputs(
                    profile_name=profile_name,
                    alpha=alpha,
                    profile=profile,
                    results_dir=results_dir,
                    pointwise_rows=pointwise_rows,
                    selected_scenarios=selected_scenarios,
                    scenario_offset=scenario_offset,
                    scenario_count=scenario_count,
                )

    outputs = _write_outputs(
        profile_name=profile_name,
        alpha=alpha,
        profile=profile,
        results_dir=results_dir,
        pointwise_rows=pointwise_rows,
        selected_scenarios=selected_scenarios,
        scenario_offset=scenario_offset,
        scenario_count=scenario_count,
    )
    return {
        "profile_name": profile_name,
        "results_dir": results_dir,
        **outputs,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run targeted simulation-based inference validation.")
    parser.add_argument("--profile", default="pilot", choices=("smoke", "pilot", "full"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed inference replications from an existing pointwise results file.",
    )
    parser.add_argument("--scenario-offset", type=int, default=0)
    parser.add_argument("--scenario-count", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    outputs = run_profile(
        profile_name=args.profile,
        results_dir=args.results_dir,
        alpha=args.alpha,
        resume=args.resume,
        scenario_offset=args.scenario_offset,
        scenario_count=args.scenario_count,
    )
    print(f"Inference profile '{args.profile}' completed.")
    print(f"Manifest written to {outputs['manifest_path']}")


if __name__ == "__main__":
    main()
