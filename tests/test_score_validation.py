import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "kan-d-iv-late" / "code"


def load_score_module():
    path = CODE_DIR / "dlate_score.py"
    spec = importlib.util.spec_from_file_location("dlate_score_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_data():
    return pd.DataFrame(
        {
            "Z": [0, 1, 0, 1],
            "W": [0, 1, 0, 1],
            "Y": [0.0, 1.0, 0.5, 1.5],
        }
    )


def sample_nuisance():
    return {
        "pi_hat": np.array([0.0, 0.5, 0.5, 1.0]),
        "p_hat_0": np.array([0.2, 0.2, 0.2, 0.2]),
        "p_hat_1": np.array([0.8, 0.8, 0.8, 0.8]),
        "mu_hat_0": np.array([[0.0], [0.2], [0.3], [1.0]]),
        "mu_hat_1": np.array([[0.6], [0.6], [0.7], [0.7]]),
    }


def test_score_validation_accepts_boundary_probabilities_before_clipping():
    score = load_score_module()
    results = score.compute_dlate_score_objects(
        sample_data(),
        sample_nuisance(),
        np.array([0.75]),
    )

    assert results["dlate"].shape == (1,)
    assert np.isfinite(results["psi_beta"]).all()


def test_score_validation_requires_data_columns():
    score = load_score_module()
    data = sample_data().drop(columns=["W"])

    with pytest.raises(KeyError, match="Missing required data columns"):
        score.compute_dlate_score_objects(data, sample_nuisance(), np.array([0.75]))


def test_score_validation_rejects_nonfinite_data_values():
    score = load_score_module()
    data = sample_data()
    data.loc[0, "Y"] = np.nan

    with pytest.raises(ValueError, match="Data column 'Y'"):
        score.compute_dlate_score_objects(data, sample_nuisance(), np.array([0.75]))


@pytest.mark.parametrize(
    "bad_y_grid",
    [
        np.array([[0.75]]),
        np.array([0.75, np.nan]),
        np.array([]),
    ],
)
def test_score_validation_rejects_malformed_y_grid(bad_y_grid):
    score = load_score_module()

    with pytest.raises(ValueError, match="y_grid"):
        score.compute_dlate_score_objects(sample_data(), sample_nuisance(), bad_y_grid)


@pytest.mark.parametrize("bad_epsilon", [0.0, -1e-6, 0.5, np.nan, np.array([1e-6, 1e-5])])
def test_score_validation_rejects_malformed_epsilon(bad_epsilon):
    score = load_score_module()

    with pytest.raises(ValueError, match="epsilon"):
        score.compute_dlate_score_objects(
            sample_data(),
            sample_nuisance(),
            np.array([0.75]),
            epsilon=bad_epsilon,
        )


def test_score_validation_rejects_vector_nuisance_shape_mismatch():
    score = load_score_module()
    nuisance = sample_nuisance()
    nuisance["pi_hat"] = np.array([[0.5], [0.5], [0.5], [0.5]])

    with pytest.raises(ValueError, match="pi_hat"):
        score.compute_dlate_score_objects(sample_data(), nuisance, np.array([0.75]))


def test_score_validation_rejects_matrix_nuisance_shape_mismatch():
    score = load_score_module()
    nuisance = sample_nuisance()
    nuisance["mu_hat_0"] = np.array([0.2, 0.2, 0.3, 0.3])

    with pytest.raises(ValueError, match="mu_hat_0"):
        score.compute_dlate_score_objects(sample_data(), nuisance, np.array([0.75]))


def test_score_validation_rejects_nonfinite_nuisance_values():
    score = load_score_module()
    nuisance = sample_nuisance()
    nuisance["p_hat_1"] = np.array([0.8, np.inf, 0.8, 0.8])

    with pytest.raises(ValueError, match="p_hat_1"):
        score.compute_dlate_score_objects(sample_data(), nuisance, np.array([0.75]))
