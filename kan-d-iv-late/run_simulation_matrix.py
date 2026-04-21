from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CODE_DIR = PROJECT_DIR / "code"
SCRIPT_PATH = CODE_DIR / "kan-d-iv-late_simulation.py"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "simulation_runs" / "matrix"

DESIGN_NAMES = ("smooth_low", "baseline", "complex_local")
INSTRUMENT_STRENGTH_NAMES = ("weak", "medium", "strong")
SAMPLE_SIZE_GRID = (500, 1000, 2000)
DEFAULT_TRUTH_SEED = 1729
RF_CONFIG_ID = "rf_core_v1"
KAN_CONFIG_ID = "kan_core_v1"
SEED_BASE_START = 20260420
SCENARIO_REQUIRED_OUTPUT_KEYS = (
    "truth",
    "summary",
    "replications",
    "diagnostics",
    "diagnostics_summary",
    "compatibility_summary",
    "manifest",
)
KAN_ABLATION_OUTPUT_KEYS = (
    "kan_ablation_summary",
    "kan_ablation_model_comparison",
    "kan_ablation_lock_decision",
)


def build_kan_config(
    *,
    steps=25,
    hidden_dim=16,
    grid_size=4,
    spline_order=3,
    lr=1e-3,
    weight_decay=1e-4,
    reg_strength=1e-4,
    min_class_count=5,
    probability_epsilon=1e-6,
):
    return {
        "steps": int(steps),
        "hidden_dim": int(hidden_dim),
        "grid_size": int(grid_size),
        "spline_order": int(spline_order),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "reg_strength": float(reg_strength),
        "min_class_count": int(min_class_count),
        "probability_epsilon": float(probability_epsilon),
    }


def build_kan_config_id(config, *, default_id="kan_core_v1"):
    normalized = build_kan_config(**config)
    if normalized == build_kan_config():
        return default_id
    reg_component = f"{normalized['reg_strength']:.0e}".replace("+0", "").replace("+", "")
    lr_component = f"{normalized['lr']:.0e}".replace("+0", "").replace("+", "")
    wd_component = f"{normalized['weight_decay']:.0e}".replace("+0", "").replace("+", "")
    return (
        "kan"
        f"_hd{normalized['hidden_dim']}"
        f"_gs{normalized['grid_size']}"
        f"_sp{normalized['spline_order']}"
        f"_st{normalized['steps']}"
        f"_lr{lr_component}"
        f"_wd{wd_component}"
        f"_reg{reg_component}"
    )

KAN_ABLATION_CONFIG_LIBRARY = (
    {"label": "kan_core_v1", "params": build_kan_config(steps=25, hidden_dim=16, grid_size=4, reg_strength=1e-4)},
    {"label": "kan_width8_v1", "params": build_kan_config(steps=25, hidden_dim=8, grid_size=4, reg_strength=1e-4)},
    {"label": "kan_width64_v1", "params": build_kan_config(steps=25, hidden_dim=64, grid_size=4, reg_strength=1e-4)},
    {"label": "kan_steps10_v1", "params": build_kan_config(steps=10, hidden_dim=16, grid_size=4, reg_strength=1e-4)},
    {"label": "kan_steps50_v1", "params": build_kan_config(steps=50, hidden_dim=16, grid_size=4, reg_strength=1e-4)},
    {"label": "kan_reg1e-5_v1", "params": build_kan_config(steps=25, hidden_dim=16, grid_size=4, reg_strength=1e-5)},
    {"label": "kan_reg1e-3_v1", "params": build_kan_config(steps=25, hidden_dim=16, grid_size=4, reg_strength=1e-3)},
    {"label": "kan_grid3_v1", "params": build_kan_config(steps=25, hidden_dim=16, grid_size=3, reg_strength=1e-4)},
    {"label": "kan_grid6_v1", "params": build_kan_config(steps=25, hidden_dim=16, grid_size=6, reg_strength=1e-4)},
)


def load_simulation_module():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    spec = importlib.util.spec_from_file_location("kan_d_iv_late_simulation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def scenario_label(scenario):
    return (
        f"{scenario['design_name']}__"
        f"{scenario['instrument_strength']}__"
        f"n{scenario['n_samples']}__"
        f"nsim{scenario['n_simulations']}"
    )


def _build_cartesian_scenarios(*, sample_sizes, strengths, n_simulations, y_points, k_folds, kan_steps, truth_sample_size):
    scenarios = []
    kan_config_id = build_kan_config_id(build_kan_config(steps=kan_steps), default_id=KAN_CONFIG_ID)
    for design_name, instrument_strength, n_samples in product(DESIGN_NAMES, strengths, sample_sizes):
        scenarios.append(
            {
                "design_name": design_name,
                "instrument_strength": instrument_strength,
                "n_samples": n_samples,
                "n_simulations": n_simulations,
                "y_points": y_points,
                "k_folds": k_folds,
                "kan_steps": kan_steps,
                "truth_sample_size": truth_sample_size,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": kan_config_id,
            }
        )
    return scenarios


def build_profile_scenarios(profile_name):
    if profile_name == "smoke":
        smoke_config_id = build_kan_config_id(build_kan_config(steps=1), default_id=KAN_CONFIG_ID)
        return [
            {
                "design_name": "smooth_low",
                "instrument_strength": "medium",
                "n_samples": 60,
                "n_simulations": 1,
                "y_points": 3,
                "k_folds": 2,
                "kan_steps": 1,
                "truth_sample_size": 3000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": smoke_config_id,
            },
            {
                "design_name": "baseline",
                "instrument_strength": "weak",
                "n_samples": 80,
                "n_simulations": 1,
                "y_points": 3,
                "k_folds": 2,
                "kan_steps": 1,
                "truth_sample_size": 3000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": smoke_config_id,
            },
            {
                "design_name": "baseline",
                "instrument_strength": "strong",
                "n_samples": 120,
                "n_simulations": 1,
                "y_points": 3,
                "k_folds": 2,
                "kan_steps": 1,
                "truth_sample_size": 3000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": smoke_config_id,
            },
            {
                "design_name": "complex_local",
                "instrument_strength": "medium",
                "n_samples": 80,
                "n_simulations": 1,
                "y_points": 3,
                "k_folds": 2,
                "kan_steps": 1,
                "truth_sample_size": 3000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": smoke_config_id,
            },
        ]

    if profile_name == "pilot":
        return _build_cartesian_scenarios(
            sample_sizes=(500, 2000),
            strengths=("weak", "strong"),
            n_simulations=10,
            y_points=8,
            k_folds=3,
            kan_steps=25,
            truth_sample_size=30000,
        )

    if profile_name == "full":
        return _build_cartesian_scenarios(
            sample_sizes=SAMPLE_SIZE_GRID,
            strengths=INSTRUMENT_STRENGTH_NAMES,
            n_simulations=50,
            y_points=10,
            k_folds=3,
            kan_steps=25,
            truth_sample_size=50000,
        )

    raise ValueError(f"Unsupported profile_name: {profile_name}")


def build_kan_ablation_scenarios(profile_name):
    if profile_name == "smoke":
        return [
            {
                "design_name": "complex_local",
                "instrument_strength": "weak",
                "n_samples": 120,
                "n_simulations": 1,
                "y_points": 3,
                "k_folds": 2,
                "kan_steps": 25,
                "truth_sample_size": 3000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": KAN_CONFIG_ID,
            }
        ]

    if profile_name == "pilot":
        return [
            {
                "design_name": "baseline",
                "instrument_strength": "weak",
                "n_samples": 2000,
                "n_simulations": 5,
                "y_points": 8,
                "k_folds": 3,
                "kan_steps": 25,
                "truth_sample_size": 30000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": KAN_CONFIG_ID,
            },
            {
                "design_name": "complex_local",
                "instrument_strength": "weak",
                "n_samples": 2000,
                "n_simulations": 5,
                "y_points": 8,
                "k_folds": 3,
                "kan_steps": 25,
                "truth_sample_size": 30000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": KAN_CONFIG_ID,
            },
        ]

    if profile_name == "full":
        return [
            {
                "design_name": "baseline",
                "instrument_strength": "weak",
                "n_samples": 2000,
                "n_simulations": 10,
                "y_points": 10,
                "k_folds": 3,
                "kan_steps": 25,
                "truth_sample_size": 50000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": KAN_CONFIG_ID,
            },
            {
                "design_name": "complex_local",
                "instrument_strength": "weak",
                "n_samples": 2000,
                "n_simulations": 10,
                "y_points": 10,
                "k_folds": 3,
                "kan_steps": 25,
                "truth_sample_size": 50000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": KAN_CONFIG_ID,
            },
            {
                "design_name": "complex_local",
                "instrument_strength": "strong",
                "n_samples": 2000,
                "n_simulations": 10,
                "y_points": 10,
                "k_folds": 3,
                "kan_steps": 25,
                "truth_sample_size": 50000,
                "truth_seed": DEFAULT_TRUTH_SEED,
                "rf_config_id": RF_CONFIG_ID,
                "kan_config_id": KAN_CONFIG_ID,
            },
        ]

    raise ValueError(f"Unsupported ablation profile_name: {profile_name}")


def build_kan_ablation_configs(profile_name):
    if profile_name == "smoke":
        labels = {"kan_core_v1", "kan_width64_v1", "kan_steps50_v1"}
    elif profile_name == "pilot":
        labels = {"kan_core_v1", "kan_width8_v1", "kan_width64_v1", "kan_steps50_v1", "kan_reg1e-3_v1"}
    elif profile_name == "full":
        labels = {item["label"] for item in KAN_ABLATION_CONFIG_LIBRARY}
    else:
        raise ValueError(f"Unsupported ablation profile_name: {profile_name}")

    configs = []
    for item in KAN_ABLATION_CONFIG_LIBRARY:
        if item["label"] in labels:
            configs.append(
                {
                    "kan_ablation_label": item["label"],
                    "kan_config_id": build_kan_config_id(item["params"]),
                    "kan_config": dict(item["params"]),
                }
            )
    return configs


def render_profile_listing(profile_name):
    scenarios = build_profile_scenarios(profile_name)
    lines = [f"Profile: {profile_name}", f"Scenario count: {len(scenarios)}"]
    for scenario in scenarios:
        lines.append(
            " - "
            f"{scenario_label(scenario)} "
            f"[design={scenario['design_name']}, "
            f"strength={scenario['instrument_strength']}, "
            f"n={scenario['n_samples']}]"
        )
    return "\n".join(lines)


def render_kan_ablation_listing(profile_name):
    scenarios = build_kan_ablation_scenarios(profile_name)
    configs = build_kan_ablation_configs(profile_name)
    lines = [
        f"KAN ablation profile: {profile_name}",
        f"Scenario count: {len(scenarios)}",
        f"Config count: {len(configs)}",
    ]
    for scenario in scenarios:
        lines.append(f" - scenario {scenario_label(scenario)}")
    for config in configs:
        lines.append(
            " - config "
            f"{config['kan_ablation_label']} "
            f"[id={config['kan_config_id']}, "
            f"hidden_dim={config['kan_config']['hidden_dim']}, "
            f"grid_size={config['kan_config']['grid_size']}, "
            f"steps={config['kan_config']['steps']}, "
            f"reg_strength={config['kan_config']['reg_strength']}]"
        )
    return "\n".join(lines)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_completed_scenario_record(scenario_dir, scenario, *, profile_name, seed_base):
    scenario_dir = Path(scenario_dir)
    manifest_paths = sorted(scenario_dir.glob("simulation_manifest_*.json"))
    if len(manifest_paths) != 1:
        return None

    manifest = _load_json(manifest_paths[0])
    expected_fields = {
        "profile_name": profile_name,
        "scenario_label": scenario_label(scenario),
        "design_name": scenario["design_name"],
        "instrument_strength": scenario["instrument_strength"],
        "n_samples": scenario["n_samples"],
        "n_simulations": scenario["n_simulations"],
        "y_points": scenario["y_points"],
        "k_folds": scenario["k_folds"],
        "kan_steps": scenario["kan_steps"],
        "truth_sample_size": scenario["truth_sample_size"],
        "truth_seed": scenario["truth_seed"],
        "seed_base": seed_base,
        "rf_config_id": scenario["rf_config_id"],
        "kan_config_id": scenario["kan_config_id"],
    }
    for key, expected_value in expected_fields.items():
        if manifest.get(key) != expected_value:
            return None

    output_files = manifest.get("output_files", {})
    for key in SCENARIO_REQUIRED_OUTPUT_KEYS:
        output_path = output_files.get(key)
        if not output_path or not Path(output_path).exists():
            return None

    return {
        **scenario,
        "seed_base": seed_base,
        "scenario_label": manifest["scenario_label"],
        "results_dir": str(scenario_dir),
        "manifest_path": str(manifest_paths[0]),
        "run_label": manifest["run_label"],
        "status": "resumed",
    }


def select_scenarios(scenarios, *, scenario_offset=0, scenario_count=None):
    if scenario_offset < 0:
        raise ValueError("scenario_offset must be non-negative")
    selected = scenarios[scenario_offset:]
    if scenario_count is not None:
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive when provided")
        selected = selected[:scenario_count]
    return selected


def discover_completed_profile_records(profile_name, profile_dir, scenarios):
    completed_records = []
    for scenario_index, scenario in enumerate(scenarios):
        seed_base = SEED_BASE_START + 1000 * scenario_index
        scenario_dir = profile_dir / scenario_label(scenario)
        record = load_completed_scenario_record(
            scenario_dir,
            scenario,
            profile_name=profile_name,
            seed_base=seed_base,
        )
        if record is not None:
            completed_records.append(record)
    return completed_records


def build_profile_model_comparison(profile_scenario_summary):
    metadata_columns = [
        "profile_name",
        "scenario_label",
        "design_name",
        "instrument_strength",
        "n_samples",
        "n_simulations",
        "y_points",
        "k_folds",
        "kan_steps",
        "truth_sample_size",
        "truth_seed",
        "seed_base",
        "rf_config_id",
        "kan_config_id",
    ]
    comparison_metrics = [
        "scenario_integrated_rmse",
        "scenario_mean_abs_error",
        "scenario_nan_rate",
        "fragile_denominator_share",
        "weak_first_stage_share",
        "mean_runtime_sec",
    ]

    rf_summary = profile_scenario_summary[profile_scenario_summary["model"] == "rf"][
        metadata_columns + comparison_metrics
    ].rename(columns={metric: f"rf_{metric}" for metric in comparison_metrics})
    kan_summary = profile_scenario_summary[profile_scenario_summary["model"] == "kan"][
        metadata_columns + comparison_metrics
    ].rename(columns={metric: f"kan_{metric}" for metric in comparison_metrics})

    comparison_df = rf_summary.merge(
        kan_summary,
        on=metadata_columns,
        how="outer",
        validate="one_to_one",
    )

    for metric in comparison_metrics:
        comparison_df[f"kan_minus_rf_{metric}"] = (
            comparison_df[f"kan_{metric}"] - comparison_df[f"rf_{metric}"]
        )

    return comparison_df.sort_values(["scenario_label"]).reset_index(drop=True)


def aggregate_profile_outputs(profile_name, profile_dir, scenario_records):
    scenario_summary_frames = []

    for scenario_record in scenario_records:
        scenario_manifest = _load_json(scenario_record["manifest_path"])
        scenario_summary_path = Path(
            scenario_manifest["output_files"].get(
                "scenario_summary",
                scenario_manifest["output_files"]["diagnostics_summary"],
            )
        )
        scenario_summary = pd.read_csv(scenario_summary_path)
        scenario_summary_frames.append(scenario_summary)

    if scenario_summary_frames:
        profile_scenario_summary = pd.concat(scenario_summary_frames, ignore_index=True)
        profile_scenario_summary = profile_scenario_summary.sort_values(
            ["scenario_label", "model"]
        ).reset_index(drop=True)
    else:
        profile_scenario_summary = pd.DataFrame()

    profile_model_comparison = build_profile_model_comparison(profile_scenario_summary)

    scenario_summary_path = profile_dir / "profile_scenario_summary.csv"
    model_comparison_path = profile_dir / "profile_model_comparison.csv"
    profile_scenario_summary.to_csv(scenario_summary_path, index=False)
    profile_model_comparison.to_csv(model_comparison_path, index=False)

    aggregate_manifest = {
        "profile_name": profile_name,
        "scenario_count": len(scenario_records),
        "aggregate_files": {
            "profile_scenario_summary": str(scenario_summary_path),
            "profile_model_comparison": str(model_comparison_path),
        },
        "scenario_labels": [record["scenario_label"] for record in scenario_records],
    }
    return {
        "profile_scenario_summary": scenario_summary_path,
        "profile_model_comparison": model_comparison_path,
        "aggregate_manifest": aggregate_manifest,
    }


def _ablation_run_key(scenario_record):
    return f"{scenario_record['scenario_label']}__{scenario_record['kan_ablation_label']}"


def aggregate_kan_ablation_outputs(profile_name, profile_dir, run_records):
    scenario_summary_frames = []

    for run_record in run_records:
        run_manifest = _load_json(run_record["manifest_path"])
        scenario_summary_path = Path(
            run_manifest["output_files"].get(
                "scenario_summary",
                run_manifest["output_files"]["diagnostics_summary"],
            )
        )
        scenario_summary = pd.read_csv(scenario_summary_path)
        scenario_summary["kan_ablation_label"] = run_record["kan_ablation_label"]
        scenario_summary["kan_ablation_config_id"] = run_record["kan_config_id"]
        scenario_summary_frames.append(scenario_summary)

    if scenario_summary_frames:
        ablation_summary = pd.concat(scenario_summary_frames, ignore_index=True)
        ablation_summary = ablation_summary.sort_values(
            ["scenario_label", "kan_ablation_label", "model"]
        ).reset_index(drop=True)
    else:
        ablation_summary = pd.DataFrame()

    comparison_metrics = [
        "scenario_integrated_rmse",
        "scenario_mean_abs_error",
        "scenario_nan_rate",
        "fragile_denominator_share",
        "weak_first_stage_share",
        "mean_runtime_sec",
    ]
    metadata_columns = [
        "profile_name",
        "scenario_label",
        "design_name",
        "instrument_strength",
        "n_samples",
        "n_simulations",
        "y_points",
        "k_folds",
        "truth_sample_size",
        "truth_seed",
        "seed_base",
        "rf_config_id",
        "kan_config_id",
        "kan_ablation_label",
        "kan_ablation_config_id",
    ]
    if ablation_summary.empty:
        ablation_comparison = pd.DataFrame()
        lock_decision = {}
    else:
        rf_summary = ablation_summary[ablation_summary["model"] == "rf"][
            metadata_columns + comparison_metrics
        ].rename(columns={metric: f"rf_{metric}" for metric in comparison_metrics})
        kan_summary = ablation_summary[ablation_summary["model"] == "kan"][
            metadata_columns + comparison_metrics
        ].rename(columns={metric: f"kan_{metric}" for metric in comparison_metrics})
        ablation_comparison = rf_summary.merge(
            kan_summary,
            on=metadata_columns,
            how="outer",
            validate="one_to_one",
        )
        for metric in comparison_metrics:
            ablation_comparison[f"kan_minus_rf_{metric}"] = (
                ablation_comparison[f"kan_{metric}"] - ablation_comparison[f"rf_{metric}"]
            )
        ablation_comparison = ablation_comparison.sort_values(
            ["scenario_label", "kan_ablation_label"]
        ).reset_index(drop=True)

        lock_table = (
            ablation_comparison.groupby(
                ["kan_ablation_label", "kan_ablation_config_id", "kan_config_id"],
                as_index=False,
            )
            .agg(
                scenario_count=("scenario_label", "nunique"),
                mean_integrated_rmse=("kan_scenario_integrated_rmse", "mean"),
                mean_abs_error=("kan_scenario_mean_abs_error", "mean"),
                mean_fragile_denominator_share=("kan_fragile_denominator_share", "mean"),
                mean_runtime_sec=("kan_mean_runtime_sec", "mean"),
            )
            .sort_values(
                [
                    "mean_integrated_rmse",
                    "mean_fragile_denominator_share",
                    "mean_runtime_sec",
                    "kan_ablation_label",
                ]
            )
            .reset_index(drop=True)
        )
        selected = lock_table.iloc[0].to_dict()
        lock_decision = {
            "selected_kan_ablation_label": selected["kan_ablation_label"],
            "selected_kan_config_id": selected["kan_config_id"],
            "selection_rule": (
                "Minimize average KAN integrated RMSE across ablation scenarios; "
                "break ties by lower fragile denominator share, then lower runtime."
            ),
            "selection_table": lock_table.to_dict(orient="records"),
        }

    summary_path = profile_dir / "kan_ablation_summary.csv"
    comparison_path = profile_dir / "kan_ablation_model_comparison.csv"
    decision_path = profile_dir / "kan_ablation_lock_decision.json"
    ablation_summary.to_csv(summary_path, index=False)
    ablation_comparison.to_csv(comparison_path, index=False)
    _write_json(decision_path, lock_decision)

    return {
        "kan_ablation_summary": summary_path,
        "kan_ablation_model_comparison": comparison_path,
        "kan_ablation_lock_decision": decision_path,
    }


def run_profile(
    profile_name="smoke",
    *,
    results_dir=DEFAULT_RESULTS_DIR,
    simulation_module=None,
    resume=False,
    scenario_offset=0,
    scenario_count=None,
):
    simulation_module = simulation_module or load_simulation_module()
    results_dir = Path(results_dir)
    profile_dir = results_dir / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_profile_scenarios(profile_name)
    selected_scenarios = select_scenarios(
        scenarios,
        scenario_offset=scenario_offset,
        scenario_count=scenario_count,
    )
    manifest_path = profile_dir / f"profile_manifest_{profile_name}.json"
    if resume and manifest_path.exists():
        manifest = _load_json(manifest_path)
        manifest["profile_name"] = profile_name
        manifest["scenario_count"] = len(scenarios)
    else:
        manifest = {
            "profile_name": profile_name,
            "scenario_count": len(scenarios),
            "scenarios": [],
        }
    scenario_dirs = []
    scenario_records_by_label = {
        record["scenario_label"]: record for record in manifest.get("scenarios", [])
    }

    for scenario_index, scenario in enumerate(scenarios):
        if scenario not in selected_scenarios:
            continue
        scenario_dir = profile_dir / scenario_label(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        seed_base = SEED_BASE_START + 1000 * scenario_index
        scenario_record = None
        if resume:
            scenario_record = load_completed_scenario_record(
                scenario_dir,
                scenario,
                profile_name=profile_name,
                seed_base=seed_base,
            )
            if scenario_record is not None:
                print(f"Resuming from existing scenario artifacts: {scenario_record['scenario_label']}")

        if scenario_record is None:
            scenario_outputs = simulation_module.main(
                results_dir=scenario_dir,
                n_simulations=scenario["n_simulations"],
                n_samples=scenario["n_samples"],
                design_name=scenario["design_name"],
                instrument_strength=scenario["instrument_strength"],
                y_points=scenario["y_points"],
                k_folds=scenario["k_folds"],
                kan_steps=scenario["kan_steps"],
                truth_sample_size=scenario["truth_sample_size"],
                truth_seed=scenario["truth_seed"],
                seed_base=seed_base,
                tag=profile_name,
                profile_name=profile_name,
                scenario_label=scenario_label(scenario),
            )
            scenario_record = {
                **scenario,
                "seed_base": seed_base,
                "scenario_label": scenario_label(scenario),
                "results_dir": str(scenario_dir),
                "manifest_path": str(scenario_outputs["output_paths"]["manifest"]),
                "run_label": scenario_outputs["run_label"],
                "status": "completed",
            }

        scenario_dirs.append(scenario_dir)
        scenario_records_by_label[scenario_record["scenario_label"]] = scenario_record
        manifest["scenarios"] = sorted(
            scenario_records_by_label.values(),
            key=lambda record: record["scenario_label"],
        )
        manifest["completed_scenario_count"] = len(
            discover_completed_profile_records(profile_name, profile_dir, scenarios)
        )
        manifest["last_selection"] = {
            "scenario_offset": scenario_offset,
            "scenario_count": scenario_count,
            "selected_scenario_labels": [scenario_label(item) for item in selected_scenarios],
        }
        _write_json(manifest_path, manifest)

    completed_records = discover_completed_profile_records(profile_name, profile_dir, scenarios)
    manifest["scenarios"] = sorted(completed_records, key=lambda record: record["scenario_label"])
    manifest["completed_scenario_count"] = len(completed_records)
    aggregate_outputs = aggregate_profile_outputs(profile_name, profile_dir, completed_records)
    manifest["aggregate_files"] = {
        "profile_scenario_summary": str(aggregate_outputs["profile_scenario_summary"]),
        "profile_model_comparison": str(aggregate_outputs["profile_model_comparison"]),
    }

    _write_json(manifest_path, manifest)

    return {
        "profile_name": profile_name,
        "profile_dir": profile_dir,
        "manifest_path": manifest_path,
        "scenarios": scenarios,
        "scenario_dirs": scenario_dirs,
        "aggregate_files": aggregate_outputs["aggregate_manifest"]["aggregate_files"],
    }


def run_kan_ablation(
    profile_name="smoke",
    *,
    results_dir=DEFAULT_RESULTS_DIR,
    simulation_module=None,
    resume=False,
    scenario_offset=0,
    scenario_count=None,
):
    simulation_module = simulation_module or load_simulation_module()
    results_dir = Path(results_dir)
    profile_dir = results_dir / "ablation_kan" / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_kan_ablation_scenarios(profile_name)
    configs = build_kan_ablation_configs(profile_name)
    selected_scenarios = select_scenarios(
        scenarios,
        scenario_offset=scenario_offset,
        scenario_count=scenario_count,
    )

    manifest_path = profile_dir / f"kan_ablation_manifest_{profile_name}.json"
    if resume and manifest_path.exists():
        manifest = _load_json(manifest_path)
        manifest["profile_name"] = profile_name
        manifest["scenario_count"] = len(scenarios)
        manifest["kan_config_count"] = len(configs)
    else:
        manifest = {
            "profile_name": profile_name,
            "scenario_count": len(scenarios),
            "kan_config_count": len(configs),
            "ablation_runs": [],
        }

    run_records_by_key = {
        _ablation_run_key(record): record for record in manifest.get("ablation_runs", [])
    }

    for scenario_index, scenario in enumerate(scenarios):
        if scenario not in selected_scenarios:
            continue

        seed_base = SEED_BASE_START + 1000 * scenario_index
        base_label = scenario_label(scenario)

        for config in configs:
            run_dir = profile_dir / base_label / config["kan_ablation_label"]
            run_dir.mkdir(parents=True, exist_ok=True)
            run_record = None

            if resume:
                resume_scenario = {
                    **scenario,
                    "kan_config_id": config["kan_config_id"],
                    "kan_steps": config["kan_config"]["steps"],
                }
                run_record = load_completed_scenario_record(
                    run_dir,
                    resume_scenario,
                    profile_name=f"{profile_name}_kan_ablation",
                    seed_base=seed_base,
                )
                if run_record is not None:
                    run_record["kan_ablation_label"] = config["kan_ablation_label"]
                    run_record["status"] = "resumed"
                    print(
                        "Resuming from existing KAN ablation artifacts: "
                        f"{base_label} / {config['kan_ablation_label']}"
                    )

            if run_record is None:
                outputs = simulation_module.main(
                    results_dir=run_dir,
                    n_simulations=scenario["n_simulations"],
                    n_samples=scenario["n_samples"],
                    design_name=scenario["design_name"],
                    instrument_strength=scenario["instrument_strength"],
                    y_points=scenario["y_points"],
                    k_folds=scenario["k_folds"],
                    kan_steps=config["kan_config"]["steps"],
                    kan_hidden_dim=config["kan_config"]["hidden_dim"],
                    kan_grid_size=config["kan_config"]["grid_size"],
                    kan_spline_order=config["kan_config"]["spline_order"],
                    kan_lr=config["kan_config"]["lr"],
                    kan_weight_decay=config["kan_config"]["weight_decay"],
                    kan_reg_strength=config["kan_config"]["reg_strength"],
                    kan_min_class_count=config["kan_config"]["min_class_count"],
                    probability_epsilon=config["kan_config"]["probability_epsilon"],
                    truth_sample_size=scenario["truth_sample_size"],
                    truth_seed=scenario["truth_seed"],
                    seed_base=seed_base,
                    tag=f"{profile_name}_kan_ablation",
                    profile_name=f"{profile_name}_kan_ablation",
                    scenario_label=base_label,
                )
                run_record = {
                    **scenario,
                    "seed_base": seed_base,
                    "scenario_label": base_label,
                    "kan_ablation_label": config["kan_ablation_label"],
                    "kan_config_id": config["kan_config_id"],
                    "results_dir": str(run_dir),
                    "manifest_path": str(outputs["output_paths"]["manifest"]),
                    "run_label": outputs["run_label"],
                    "status": "completed",
                }

            run_records_by_key[_ablation_run_key(run_record)] = run_record
            manifest["ablation_runs"] = sorted(
                run_records_by_key.values(),
                key=lambda record: (record["scenario_label"], record["kan_ablation_label"]),
            )
            manifest["completed_run_count"] = len(manifest["ablation_runs"])
            manifest["last_selection"] = {
                "scenario_offset": scenario_offset,
                "scenario_count": scenario_count,
                "selected_scenario_labels": [scenario_label(item) for item in selected_scenarios],
                "selected_kan_ablation_labels": [item["kan_ablation_label"] for item in configs],
            }
            _write_json(manifest_path, manifest)

    completed_records = list(
        sorted(
            run_records_by_key.values(),
            key=lambda record: (record["scenario_label"], record["kan_ablation_label"]),
        )
    )
    aggregate_files = aggregate_kan_ablation_outputs(profile_name, profile_dir, completed_records)
    manifest["ablation_runs"] = completed_records
    manifest["completed_run_count"] = len(completed_records)
    manifest["aggregate_files"] = {key: str(value) for key, value in aggregate_files.items()}
    _write_json(manifest_path, manifest)

    return {
        "profile_name": profile_name,
        "profile_dir": profile_dir,
        "manifest_path": manifest_path,
        "scenarios": scenarios,
        "kan_configs": configs,
        "aggregate_files": {key: str(value) for key, value in aggregate_files.items()},
    }


def build_parser():
    parser = argparse.ArgumentParser(description="List or run the Phase 3 simulation scenario matrix.")
    parser.add_argument("--profile", default="smoke", choices=("smoke", "pilot", "full"))
    parser.add_argument("--ablation", choices=("kan",))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--list", action="store_true", help="List scenarios for the selected profile without running them.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-scenario manifests when present instead of rerunning those scenarios.",
    )
    parser.add_argument("--scenario-offset", type=int, default=0)
    parser.add_argument("--scenario-count", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if args.list:
        if args.ablation == "kan":
            print(render_kan_ablation_listing(args.profile))
        else:
            print(render_profile_listing(args.profile))
        return

    if args.ablation == "kan":
        outputs = run_kan_ablation(
            profile_name=args.profile,
            results_dir=args.results_dir,
            resume=args.resume,
            scenario_offset=args.scenario_offset,
            scenario_count=args.scenario_count,
        )
        print(f"KAN ablation profile '{args.profile}' completed.")
    else:
        outputs = run_profile(
            profile_name=args.profile,
            results_dir=args.results_dir,
            resume=args.resume,
            scenario_offset=args.scenario_offset,
            scenario_count=args.scenario_count,
        )
        print(f"Profile '{args.profile}' completed.")
    print(f"Manifest written to {outputs['manifest_path']}")


if __name__ == "__main__":
    main()
