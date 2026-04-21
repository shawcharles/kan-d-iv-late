import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "kan-d-iv-late" / "code"
DATA_PATH = ROOT / "kan-d-iv-late" / "data" / "pension.csv"
PROJECT_DIR = ROOT / "kan-d-iv-late"


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


def test_empirical_main_writes_expected_artifacts(tmp_path, monkeypatch):
    empirical = load_module("kan-d-iv-late_empirical_application.py", "empirical_smoke")

    def fake_predict(train_features, train_labels, test_features, **kwargs):
        baseline = 0.5 if len(train_labels) == 0 else float(np.mean(train_labels))
        return np.full(len(test_features), baseline, dtype=float)

    monkeypatch.setattr(empirical, "fit_binary_kan_predict", fake_predict)

    empirical.main(
        data_path=DATA_PATH,
        results_dir=tmp_path,
        y_grid_points=3,
        k_folds=2,
        kan_steps=1,
    )

    results_csv = tmp_path / "empirical_kan-d-iv-late_results.csv"
    plot_path = tmp_path / "empirical_kan-d-iv-late_plot.png"

    assert results_csv.exists()
    assert plot_path.exists()

    results = pd.read_csv(results_csv)
    assert list(results.columns) == ["y_value", "dlate_estimate"]
    assert len(results) == 3


def test_empirical_model_comparison_builder_reports_gap_metrics():
    wrapper = load_module(PROJECT_DIR / "run_empirical.py", "run_empirical_wrapper")

    curves = pd.DataFrame(
        {
            "spec_label": ["core_raw_30", "core_raw_30", "core_raw_30", "core_raw_30"],
            "model": ["kan", "kan", "rf", "rf"],
            "y_value": [0.0, 1.0, 0.0, 1.0],
            "dlate_estimate": [0.5, -0.2, 0.1, 0.3],
        }
    )
    comparison = wrapper.build_empirical_model_comparison(curves)

    assert len(comparison) == 1
    assert comparison.loc[0, "spec_label"] == "core_raw_30"
    assert comparison.loc[0, "max_abs_gap"] == 0.5
