import json
from pathlib import Path

import numpy as np
import pandas as pd

from kan_test_helpers import ensure_stub_efficient_kan
from provenance_assertions import assert_manifest_provenance

ensure_stub_efficient_kan()

from kan_d_iv_late import simulation


ROOT = Path(__file__).resolve().parents[1]


def test_simulation_main_writes_expected_artifacts(tmp_path, monkeypatch):
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

    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert_manifest_provenance(manifest)

    results = pd.read_csv(results_csv)
    assert list(results.columns) == ["y", "kan_avg_bias", "kan_rmse", "rf_avg_bias", "rf_rmse"]
    assert len(results) == 2
