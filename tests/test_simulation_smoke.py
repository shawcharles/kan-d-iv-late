import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "kan-d-iv-late" / "code"


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


def load_module(filename, module_name):
    ensure_stub_efficient_kan()
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    path = CODE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulation_main_writes_expected_artifacts(tmp_path, monkeypatch):
    simulation = load_module("kan-d-iv-late_simulation.py", "simulation_smoke")

    def fake_predict(train_features, train_labels, test_features, **kwargs):
        baseline = 0.5 if len(train_labels) == 0 else float(np.mean(train_labels))
        return np.full(len(test_features), baseline, dtype=float)

    monkeypatch.setattr(simulation, "fit_binary_kan_predict", fake_predict)

    simulation.main(
        results_dir=tmp_path,
        n_simulations=1,
        n_samples=20,
        y_points=2,
        k_folds=2,
        kan_steps=1,
    )

    results_csv = tmp_path / "simulation_results.csv"
    manifest_files = list(tmp_path.glob("simulation_manifest_*.json"))
    truth_files = list(tmp_path.glob("simulation_truth_*.csv"))
    diagnostics_files = list(tmp_path.glob("simulation_diagnostics_*.csv"))
    summary_files = list(tmp_path.glob("simulation_summary_*.csv"))

    assert results_csv.exists()
    assert manifest_files
    assert truth_files
    assert diagnostics_files
    assert summary_files

    results = pd.read_csv(results_csv)
    assert list(results.columns) == ["y", "kan_avg_bias", "kan_rmse", "rf_avg_bias", "rf_rmse"]
    assert len(results) == 2
