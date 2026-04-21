from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CODE_DIR = PROJECT_DIR / "code"
SCRIPT_PATH = CODE_DIR / "kan-d-iv-late_empirical_application.py"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "empirical_runs"


def load_module():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    spec = importlib.util.spec_from_file_location("kan_d_iv_late_empirical_application", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_empirical_specs(module, profile_name, *, y_points, folds, kan_steps):
    core_kan_config = module.build_kan_config(steps=kan_steps)
    core_rf_config = module.build_rf_config()

    specs = [
        {
            "label": "core_raw_30",
            "models": ("kan", "rf"),
            "preprocess_mode": "raw",
            "y_grid_points": y_points,
            "quantile_bounds": (0.01, 0.99),
            "trim_bounds": None,
            "trim_model_type": None,
            "k_folds": folds,
            "kan_config": core_kan_config,
            "rf_config": core_rf_config,
        }
    ]

    if profile_name == "core":
        return specs

    specs.extend(
        [
            {
                "label": "grid_20_raw",
                "models": ("kan", "rf"),
                "preprocess_mode": "raw",
                "y_grid_points": 20,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": folds,
                "kan_config": core_kan_config,
                "rf_config": core_rf_config,
            },
            {
                "label": "grid_40_raw",
                "models": ("kan", "rf"),
                "preprocess_mode": "raw",
                "y_grid_points": 40,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": folds,
                "kan_config": core_kan_config,
                "rf_config": core_rf_config,
            },
            {
                "label": "standardized_30",
                "models": ("kan", "rf"),
                "preprocess_mode": "standardized",
                "y_grid_points": y_points,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": folds,
                "kan_config": core_kan_config,
                "rf_config": core_rf_config,
            },
            {
                "label": "trim_rf_05_95",
                "models": ("kan", "rf"),
                "preprocess_mode": "raw",
                "y_grid_points": y_points,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": (0.05, 0.95),
                "trim_model_type": "rf",
                "k_folds": folds,
                "kan_config": core_kan_config,
                "rf_config": core_rf_config,
            },
            {
                "label": "kan_folds10",
                "models": ("kan",),
                "preprocess_mode": "raw",
                "y_grid_points": y_points,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": 10,
                "kan_config": core_kan_config,
                "rf_config": core_rf_config,
            },
            {
                "label": "kan_steps50",
                "models": ("kan",),
                "preprocess_mode": "raw",
                "y_grid_points": y_points,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": folds,
                "kan_config": module.build_kan_config(steps=50),
                "rf_config": core_rf_config,
            },
            {
                "label": "kan_hidden64",
                "models": ("kan",),
                "preprocess_mode": "raw",
                "y_grid_points": y_points,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": folds,
                "kan_config": module.build_kan_config(steps=kan_steps, hidden_dim=64),
                "rf_config": core_rf_config,
            },
            {
                "label": "rf_trees300_leaf5",
                "models": ("rf",),
                "preprocess_mode": "raw",
                "y_grid_points": y_points,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": folds,
                "kan_config": core_kan_config,
                "rf_config": module.build_rf_config(n_estimators=300, min_samples_leaf=5),
            },
        ]
    )

    if profile_name == "robustness":
        return specs

    if profile_name == "paper":
        specs.append(
            {
                "label": "kan_steps200",
                "models": ("kan",),
                "preprocess_mode": "raw",
                "y_grid_points": y_points,
                "quantile_bounds": (0.01, 0.99),
                "trim_bounds": None,
                "trim_model_type": None,
                "k_folds": folds,
                "kan_config": module.build_kan_config(steps=200),
                "rf_config": core_rf_config,
            }
        )
        return specs

    raise ValueError(f"Unsupported empirical profile: {profile_name}")


def apply_common_trim(module, data, x_cols, spec):
    trim_bounds = spec.get("trim_bounds")
    if trim_bounds is None:
        return data.reset_index(drop=True), {
            "trim_applied": False,
            "trim_model_type": None,
            "trim_lower": np.nan,
            "trim_upper": np.nan,
            "trimmed_observation_count": 0,
        }

    lower, upper = trim_bounds
    pi_hat = module.estimate_instrument_propensity_empirical(
        data,
        x_cols,
        model_type=spec["trim_model_type"],
        k_folds=spec["k_folds"],
        kan_steps=spec["kan_config"]["steps"],
        kan_config=spec["kan_config"],
        rf_config=spec["rf_config"],
    )
    keep_mask = (pi_hat >= lower) & (pi_hat <= upper)
    trimmed = data.loc[keep_mask].reset_index(drop=True)
    trim_metadata = {
        "trim_applied": True,
        "trim_model_type": spec["trim_model_type"],
        "trim_lower": lower,
        "trim_upper": upper,
        "trimmed_observation_count": int(np.sum(~keep_mask)),
    }
    return trimmed, trim_metadata


def run_empirical_spec(module, raw_data, x_cols, spec):
    processed_data, preprocess_metadata = module.preprocess_empirical_data(
        raw_data,
        x_cols,
        preprocess_mode=spec["preprocess_mode"],
    )
    trimmed_data, trim_metadata = apply_common_trim(module, processed_data, x_cols, spec)
    y_grid = module.build_empirical_y_grid(
        trimmed_data,
        y_grid_points=spec["y_grid_points"],
        quantile_bounds=spec["quantile_bounds"],
    )

    curve_frames = []
    diagnostics_rows = []
    balance_frames = []

    for model_type in spec["models"]:
        runtime_start = time.perf_counter()
        outputs = module.run_empirical_model(
            trimmed_data,
            x_cols,
            y_grid,
            model_type=model_type,
            k_folds=spec["k_folds"],
            kan_steps=spec["kan_config"]["steps"],
            kan_config=spec["kan_config"],
            rf_config=spec["rf_config"],
        )
        runtime_sec = float(time.perf_counter() - runtime_start)

        curve_df = outputs["curve"].copy()
        curve_df["spec_label"] = spec["label"]
        curve_df["preprocess_mode"] = spec["preprocess_mode"]
        curve_df["n_obs"] = len(trimmed_data)
        curve_frames.append(curve_df)

        diagnostics_rows.append(
            {
                "spec_label": spec["label"],
                "model": model_type,
                "model_config_id": outputs["diagnostics"]["model_config_id"],
                "preprocess_mode": spec["preprocess_mode"],
                "y_grid_points": spec["y_grid_points"],
                "k_folds": spec["k_folds"],
                "n_obs": len(trimmed_data),
                "runtime_sec": runtime_sec,
                "kan_config_id": module.build_kan_config_id(spec["kan_config"]),
                "rf_config_id": module.build_rf_config_id(spec["rf_config"]),
                **preprocess_metadata,
                **trim_metadata,
                **outputs["diagnostics"],
            }
        )

        balance_df = outputs["balance"].copy()
        balance_df["spec_label"] = spec["label"]
        balance_df["model"] = model_type
        balance_frames.append(balance_df)

    return {
        "curves": pd.concat(curve_frames, ignore_index=True),
        "diagnostics": pd.DataFrame(diagnostics_rows),
        "balance": pd.concat(balance_frames, ignore_index=True),
    }


def build_empirical_model_comparison(curves_df):
    comparison_rows = []
    for spec_label, spec_curves in curves_df.groupby("spec_label"):
        models = set(spec_curves["model"].unique().tolist())
        if {"kan", "rf"} - models:
            continue
        kan_curve = spec_curves[spec_curves["model"] == "kan"][["y_value", "dlate_estimate"]].rename(
            columns={"dlate_estimate": "kan_dlate_estimate"}
        )
        rf_curve = spec_curves[spec_curves["model"] == "rf"][["y_value", "dlate_estimate"]].rename(
            columns={"dlate_estimate": "rf_dlate_estimate"}
        )
        merged = kan_curve.merge(rf_curve, on="y_value", how="inner")
        gap = merged["kan_dlate_estimate"] - merged["rf_dlate_estimate"]
        lower_tail_count = max(1, int(np.ceil(len(merged) * 0.1)))
        comparison_rows.append(
            {
                "spec_label": spec_label,
                "mean_abs_gap": float(np.mean(np.abs(gap))),
                "max_abs_gap": float(np.max(np.abs(gap))),
                "mean_signed_gap": float(np.mean(gap)),
                "lower_tail_mean_signed_gap": float(np.mean(gap.iloc[:lower_tail_count])),
                "sign_disagreement_share": float(
                    np.mean(np.sign(merged["kan_dlate_estimate"]) != np.sign(merged["rf_dlate_estimate"]))
                ),
            }
        )
    return pd.DataFrame(comparison_rows)


def write_core_compatibility_assets(curves_df, results_dir):
    core_curves = curves_df[curves_df["spec_label"] == "core_raw_30"].copy()
    if core_curves.empty:
        return {}

    compatibility_paths = {}
    for model_type, output_stem, ylabel in (
        ("kan", "empirical_kan-d-iv-late", "KAN-D-LATE(y)"),
        ("rf", "empirical_rf-d-iv-late", "RF-D-LATE(y)"),
    ):
        curve_df = core_curves[core_curves["model"] == model_type][["y_value", "dlate_estimate"]].copy()
        if curve_df.empty:
            continue
        csv_path = results_dir / f"{output_stem}_results.csv"
        plot_path = results_dir / f"{output_stem}_plot.png"
        curve_df.to_csv(csv_path, index=False)
        plt.figure(figsize=(10, 6))
        plt.plot(curve_df["y_value"], curve_df["dlate_estimate"], marker="o", linestyle="-")
        plt.title(f"Estimated Distributional LATE ({model_type.upper()}) - Pension Data")
        plt.xlabel("Net Financial Assets (y)")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        compatibility_paths[f"{model_type}_csv"] = str(csv_path)
        compatibility_paths[f"{model_type}_plot"] = str(plot_path)

    if {"kan", "rf"}.issubset(set(core_curves["model"].unique().tolist())):
        comparison_plot_path = results_dir / "empirical_dlate_comparison.png"
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        for axis, model_type, title in (
            (axes[0], "kan", "KAN-based D-IV-LATE"),
            (axes[1], "rf", "Random Forest-based D-IV-LATE"),
        ):
            curve_df = core_curves[core_curves["model"] == model_type]
            axis.plot(curve_df["y_value"], curve_df["dlate_estimate"], marker="o", linestyle="-")
            axis.set_title(title)
            axis.set_ylabel(f"{model_type.upper()}-D-LATE(y)")
            axis.grid(True)
        axes[1].set_xlabel("Net Financial Assets (y)")
        fig.tight_layout()
        fig.savefig(comparison_plot_path)
        plt.close(fig)
        compatibility_paths["comparison_plot"] = str(comparison_plot_path)

    return compatibility_paths


def build_parser():
    parser = argparse.ArgumentParser(description="Run the canonical empirical KAN-D-IV-LATE evidence package.")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "pension.csv")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--profile", default="core", choices=("core", "robustness", "paper"))
    parser.add_argument("--y-points", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--kan-steps", type=int, default=100)
    return parser


def main():
    args = build_parser().parse_args()
    module = load_module()

    results_dir = args.results_dir / args.profile
    results_dir.mkdir(parents=True, exist_ok=True)

    data, x_cols = module.load_and_prepare_data(csv_path=args.data)
    if data.empty or len(data) < 100:
        raise SystemExit("Empirical data loading failed or yielded insufficient observations.")

    specs = build_empirical_specs(
        module,
        args.profile,
        y_points=args.y_points,
        folds=args.folds,
        kan_steps=args.kan_steps,
    )

    curve_frames = []
    diagnostics_frames = []
    balance_frames = []

    for spec in specs:
        print(f"Running empirical spec: {spec['label']}")
        outputs = run_empirical_spec(module, data, x_cols, spec)
        curve_frames.append(outputs["curves"])
        diagnostics_frames.append(outputs["diagnostics"])
        balance_frames.append(outputs["balance"])

    curves_df = pd.concat(curve_frames, ignore_index=True)
    diagnostics_df = pd.concat(diagnostics_frames, ignore_index=True)
    balance_df = pd.concat(balance_frames, ignore_index=True)
    comparison_df = build_empirical_model_comparison(curves_df)

    curves_path = results_dir / "empirical_curves.csv"
    diagnostics_path = results_dir / "empirical_diagnostics.csv"
    balance_path = results_dir / "empirical_balance.csv"
    comparison_path = results_dir / "empirical_model_comparison.csv"

    curves_df.to_csv(curves_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)
    balance_df.to_csv(balance_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    compatibility_assets = write_core_compatibility_assets(curves_df, results_dir)

    manifest = {
        "profile": args.profile,
        "data_path": str(args.data),
        "spec_labels": [spec["label"] for spec in specs],
        "output_files": {
            "empirical_curves": str(curves_path),
            "empirical_diagnostics": str(diagnostics_path),
            "empirical_balance": str(balance_path),
            "empirical_model_comparison": str(comparison_path),
            **compatibility_assets,
        },
    }
    manifest_path = results_dir / f"empirical_manifest_{args.profile}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Empirical profile '{args.profile}' completed.")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
