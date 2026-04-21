import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "kan-d-iv-late"
CODE_DIR = PROJECT_DIR / "code"


def ensure_stub_efficient_kan():
    if "efficient_kan" in sys.modules:
        return

    module = types.ModuleType("efficient_kan")

    class DummyKAN(torch.nn.Module):
        def __init__(self, layers_hidden, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(layers_hidden[0], layers_hidden[-1])

        def forward(self, x):
            return self.linear(x)

        def regularization_loss(self, *args, **kwargs):
            return torch.tensor(0.0)

    module.KAN = DummyKAN
    sys.modules["efficient_kan"] = module


def load_module(path, module_name):
    ensure_stub_efficient_kan()
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_profile_enumerates_crossed_phase3_matrix():
    runner = load_module(PROJECT_DIR / "run_simulation_matrix.py", "simulation_matrix_listing")
    scenarios = runner.build_profile_scenarios("full")

    assert len(scenarios) == 27
    assert {scenario["design_name"] for scenario in scenarios} == {"smooth_low", "baseline", "complex_local"}
    assert {scenario["instrument_strength"] for scenario in scenarios} == {"weak", "medium", "strong"}
    assert {scenario["n_samples"] for scenario in scenarios} == {500, 1000, 2000}


def test_matrix_runner_smoke_profile_writes_scenario_artifacts(tmp_path, monkeypatch):
    simulation = load_module(CODE_DIR / "kan-d-iv-late_simulation.py", "simulation_matrix_smoke")
    runner = load_module(PROJECT_DIR / "run_simulation_matrix.py", "simulation_matrix_runner_smoke")

    def fake_predict(train_features, train_labels, test_features, **kwargs):
        baseline = 0.5 if len(train_labels) == 0 else float(np.mean(train_labels))
        return np.full(len(test_features), baseline, dtype=float)

    monkeypatch.setattr(simulation, "fit_binary_kan_predict", fake_predict)

    outputs = runner.run_profile(profile_name="smoke", results_dir=tmp_path, simulation_module=simulation)

    assert outputs["manifest_path"].exists()
    assert len(outputs["scenario_dirs"]) == len(runner.build_profile_scenarios("smoke"))
    assert Path(outputs["aggregate_files"]["profile_scenario_summary"]).exists()
    assert Path(outputs["aggregate_files"]["profile_model_comparison"]).exists()

    manifest = json.loads(outputs["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["profile_name"] == "smoke"
    assert manifest["scenario_count"] == len(outputs["scenario_dirs"])
    assert "aggregate_files" in manifest

    for scenario_dir in outputs["scenario_dirs"]:
        assert (scenario_dir / "simulation_results.csv").exists()

    scenario_summary = np.genfromtxt(
        outputs["aggregate_files"]["profile_scenario_summary"],
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert "scenario_integrated_rmse" in scenario_summary.dtype.names
    assert "mean_runtime_sec" in scenario_summary.dtype.names


def test_matrix_runner_resume_skips_completed_scenarios(tmp_path, monkeypatch):
    simulation = load_module(CODE_DIR / "kan-d-iv-late_simulation.py", "simulation_matrix_resume")
    runner = load_module(PROJECT_DIR / "run_simulation_matrix.py", "simulation_matrix_runner_resume")

    def fake_predict(train_features, train_labels, test_features, **kwargs):
        baseline = 0.5 if len(train_labels) == 0 else float(np.mean(train_labels))
        return np.full(len(test_features), baseline, dtype=float)

    monkeypatch.setattr(simulation, "fit_binary_kan_predict", fake_predict)
    first_outputs = runner.run_profile(profile_name="smoke", results_dir=tmp_path, simulation_module=simulation)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("simulation.main should not be called when resuming completed scenarios")

    monkeypatch.setattr(simulation, "main", fail_if_called)
    resumed_outputs = runner.run_profile(
        profile_name="smoke",
        results_dir=tmp_path,
        simulation_module=simulation,
        resume=True,
    )

    assert resumed_outputs["manifest_path"].exists()
    assert first_outputs["manifest_path"] == resumed_outputs["manifest_path"]

    manifest = json.loads(resumed_outputs["manifest_path"].read_text(encoding="utf-8"))
    assert {scenario["status"] for scenario in manifest["scenarios"]} == {"resumed"}


def test_matrix_runner_kan_ablation_smoke_writes_aggregate_artifacts(tmp_path, monkeypatch):
    simulation = load_module(CODE_DIR / "kan-d-iv-late_simulation.py", "simulation_matrix_ablation")
    runner = load_module(PROJECT_DIR / "run_simulation_matrix.py", "simulation_matrix_runner_ablation")

    def fake_predict(train_features, train_labels, test_features, **kwargs):
        baseline = 0.5 if len(train_labels) == 0 else float(np.mean(train_labels))
        return np.full(len(test_features), baseline, dtype=float)

    monkeypatch.setattr(simulation, "fit_binary_kan_predict", fake_predict)
    outputs = runner.run_kan_ablation(
        profile_name="smoke",
        results_dir=tmp_path,
        simulation_module=simulation,
    )

    manifest = json.loads(outputs["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["profile_name"] == "smoke"
    assert manifest["completed_run_count"] == 3
    assert Path(outputs["aggregate_files"]["kan_ablation_summary"]).exists()
    assert Path(outputs["aggregate_files"]["kan_ablation_model_comparison"]).exists()
    assert Path(outputs["aggregate_files"]["kan_ablation_lock_decision"]).exists()


def test_matrix_runner_kan_ablation_resume_skips_completed_step_variants(tmp_path, monkeypatch):
    simulation = load_module(CODE_DIR / "kan-d-iv-late_simulation.py", "simulation_matrix_ablation_resume")
    runner = load_module(PROJECT_DIR / "run_simulation_matrix.py", "simulation_matrix_runner_ablation_resume")

    def fake_predict(train_features, train_labels, test_features, **kwargs):
        baseline = 0.5 if len(train_labels) == 0 else float(np.mean(train_labels))
        return np.full(len(test_features), baseline, dtype=float)

    monkeypatch.setattr(simulation, "fit_binary_kan_predict", fake_predict)
    first_outputs = runner.run_kan_ablation(
        profile_name="smoke",
        results_dir=tmp_path,
        simulation_module=simulation,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("simulation.main should not be called when resuming completed KAN ablation runs")

    monkeypatch.setattr(simulation, "main", fail_if_called)
    resumed_outputs = runner.run_kan_ablation(
        profile_name="smoke",
        results_dir=tmp_path,
        simulation_module=simulation,
        resume=True,
    )

    assert resumed_outputs["manifest_path"].exists()
    assert first_outputs["manifest_path"] == resumed_outputs["manifest_path"]

    manifest = json.loads(resumed_outputs["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["completed_run_count"] == len(runner.build_kan_ablation_configs("smoke"))
    assert {run["status"] for run in manifest["ablation_runs"]} == {"resumed"}
