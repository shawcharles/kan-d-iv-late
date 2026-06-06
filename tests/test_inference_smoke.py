import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance_assertions import assert_manifest_provenance

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "kan-d-iv-late" / "code"


def load_module(filename, module_name):
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    path = CODE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
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
        "pi_hat": np.array([0.5, 0.5, 0.5, 0.5]),
        "p_hat_0": np.array([0.2, 0.2, 0.2, 0.2]),
        "p_hat_1": np.array([0.8, 0.8, 0.8, 0.8]),
        "mu_hat_0": np.array([[0.2], [0.2], [0.3], [0.3]]),
        "mu_hat_1": np.array([[0.6], [0.6], [0.7], [0.7]]),
    }


def test_asymptotic_inference_returns_expected_shapes():
    inference = load_module("dlate_inference.py", "dlate_inference_asymptotic")
    results = inference.dlate_asymptotic_inference(
        sample_data(),
        sample_nuisance(),
        np.array([0.75]),
    )

    assert results["point_estimates"].shape == (1,)
    assert results["standard_errors"].shape == (1,)
    assert results["ci_lower"].shape == (1,)
    assert results["ci_upper"].shape == (1,)


def test_bootstrap_inference_returns_expected_shapes():
    inference = load_module("dlate_inference.py", "dlate_inference_bootstrap")

    def nuisance_estimator(data, y_grid):
        return sample_nuisance()

    results = inference.bootstrap_dlate_inference(
        sample_data(),
        np.array([0.75]),
        nuisance_estimator,
        n_bootstrap=4,
        random_state=123,
    )

    assert results["point_estimates"].shape == (1,)
    assert results["ci_lower"].shape == (1,)
    assert results["ci_upper"].shape == (1,)
    assert results["bootstrap_estimates"].shape == (4, 1)
    assert results["bootstrap_mean_estimate"].shape == (1,)


def test_bootstrap_inference_uses_supplied_original_point_estimates():
    inference = load_module("dlate_inference.py", "dlate_inference_bootstrap_supplied")
    supplied_point_estimates = np.array([123.0])

    def nuisance_estimator(data, y_grid):
        return sample_nuisance()

    results = inference.bootstrap_dlate_inference(
        sample_data(),
        np.array([0.75]),
        nuisance_estimator,
        point_estimates=supplied_point_estimates,
        n_bootstrap=4,
        random_state=123,
    )

    assert np.array_equal(results["point_estimates"], supplied_point_estimates)
    assert results["bootstrap_mean_estimate"].shape == (1,)


def test_bootstrap_inference_computes_original_point_estimates_once_when_omitted():
    inference = load_module("dlate_inference.py", "dlate_inference_bootstrap_original")
    calls = []

    def nuisance_estimator(data, y_grid):
        calls.append(len(data))
        return sample_nuisance()

    results = inference.bootstrap_dlate_inference(
        sample_data(),
        np.array([0.75]),
        nuisance_estimator,
        n_bootstrap=3,
        random_state=123,
    )
    asymptotic = inference.dlate_asymptotic_inference(
        sample_data(),
        sample_nuisance(),
        np.array([0.75]),
    )

    assert calls == [4, 4, 4, 4]
    assert np.allclose(results["point_estimates"], asymptotic["point_estimates"])


def test_bootstrap_inference_rejects_supplied_point_estimate_shape_mismatch():
    inference = load_module("dlate_inference.py", "dlate_inference_bootstrap_bad_point")

    def nuisance_estimator(data, y_grid):
        return sample_nuisance()

    with pytest.raises(ValueError, match="point_estimates"):
        inference.bootstrap_dlate_inference(
            sample_data(),
            np.array([0.75]),
            nuisance_estimator,
            point_estimates=np.array([1.0, 2.0]),
            n_bootstrap=1,
        )


def test_bootstrap_inference_rejects_malformed_grid_with_supplied_point_estimates():
    inference = load_module("dlate_inference.py", "dlate_inference_bootstrap_bad_grid")

    def nuisance_estimator(data, y_grid):
        return sample_nuisance()

    with pytest.raises(ValueError, match="y_grid"):
        inference.bootstrap_dlate_inference(
            sample_data(),
            0.75,
            nuisance_estimator,
            point_estimates=np.array([1.0]),
            n_bootstrap=1,
        )


def test_inference_runner_smoke_writes_checkpointed_outputs_and_resumes(tmp_path):
    seen_kan_steps = []

    def fake_estimate_nuisance_functions(*args, **kwargs):
        seen_kan_steps.append(kwargs["kan_steps"])
        return sample_nuisance()

    runner = load_module("../run_inference_validation.py", "run_inference_validation_smoke")

    fake_simulation = types.SimpleNamespace(
        PROBABILITY_EPSILON=1e-6,
        build_truth_bundle=lambda **kwargs: {
            "truth_df": pd.DataFrame({"y": [0.0, 1.0], "true_dlate": [0.25, 0.75]})
        },
        build_kan_config=lambda **kwargs: {"steps": kwargs.get("steps", 1), "hidden_dim": kwargs.get("hidden_dim", 16)},
        build_kan_config_id=lambda config: f"kan_hd{config['hidden_dim']}_st{config['steps']}",
        generate_dlate_data=lambda **kwargs: (sample_data(), None),
        estimate_nuisance_functions=fake_estimate_nuisance_functions,
    )
    asymptotic_point_estimates = np.array([0.2, 0.8])

    def fake_bootstrap_dlate_inference(*args, **kwargs):
        assert np.array_equal(kwargs["point_estimates"], asymptotic_point_estimates)
        return {
            "point_estimates": kwargs["point_estimates"],
            "ci_lower": np.array([0.12, 0.68]),
            "ci_upper": np.array([0.32, 0.88]),
        }

    fake_inference = types.SimpleNamespace(
        dlate_asymptotic_inference=lambda *args, **kwargs: {
            "point_estimates": asymptotic_point_estimates,
            "ci_lower": np.array([0.1, 0.7]),
            "ci_upper": np.array([0.3, 0.9]),
            "mean_psi_beta": 0.5,
            "near_zero_denominator": False,
        },
        bootstrap_dlate_inference=fake_bootstrap_dlate_inference,
        summarize_interval_coverage=lambda point, lower, upper, truth: {
            "covers": np.array([True, False]),
            "widths": upper - lower,
        },
    )

    outputs = runner.run_profile(
        profile_name="smoke",
        results_dir=tmp_path,
        kan_steps=7,
        simulation_module=fake_simulation,
        inference_module=fake_inference,
    )

    manifest = json.loads(outputs["manifest_path"].read_text(encoding="utf-8"))
    pointwise = pd.read_csv(outputs["pointwise_path"])
    summary = pd.read_csv(outputs["summary_path"])

    assert manifest["profile"] == "smoke"
    assert manifest["kan_steps"] == 7
    assert manifest["kan_config_label"] is None
    assert manifest["kan_config_id"] == "kan_hd16_st7"
    assert manifest["completed_replication_count"] == 2
    assert_manifest_provenance(manifest)
    assert pointwise.shape[0] == 4
    assert set(summary["model"]) == {"kan", "rf"}
    assert seen_kan_steps == [7, 7]

    def fail_estimation(*args, **kwargs):
        raise AssertionError("resume should skip completed inference replications")

    fake_simulation.estimate_nuisance_functions = fail_estimation
    resumed = runner.run_profile(
        profile_name="smoke",
        results_dir=tmp_path,
        kan_steps=7,
        simulation_module=fake_simulation,
        inference_module=fake_inference,
        resume=True,
    )

    resumed_manifest = json.loads(resumed["manifest_path"].read_text(encoding="utf-8"))
    assert resumed_manifest["completed_replication_count"] == 2


def test_inference_runner_accepts_labeled_kan_config(tmp_path):
    seen_kan_configs = []

    def fake_estimate_nuisance_functions(*args, **kwargs):
        seen_kan_configs.append(kwargs["kan_config"])
        return sample_nuisance()

    runner = load_module("../run_inference_validation.py", "run_inference_validation_labeled_config")

    fake_simulation = types.SimpleNamespace(
        PROBABILITY_EPSILON=1e-6,
        build_truth_bundle=lambda **kwargs: {
            "truth_df": pd.DataFrame({"y": [0.0], "true_dlate": [0.25]})
        },
        build_kan_config=lambda **kwargs: {"steps": kwargs.get("steps", 1), "hidden_dim": kwargs.get("hidden_dim", 16)},
        build_kan_config_id=lambda config: f"kan_hd{config['hidden_dim']}_st{config['steps']}",
        generate_dlate_data=lambda **kwargs: (sample_data(), None),
        estimate_nuisance_functions=fake_estimate_nuisance_functions,
    )
    fake_inference = types.SimpleNamespace(
        dlate_asymptotic_inference=lambda *args, **kwargs: {
            "point_estimates": np.array([0.2]),
            "ci_lower": np.array([0.1]),
            "ci_upper": np.array([0.3]),
            "mean_psi_beta": 0.5,
            "near_zero_denominator": False,
        },
        bootstrap_dlate_inference=lambda *args, **kwargs: {
            "point_estimates": np.array([0.2]),
            "ci_lower": np.array([0.1]),
            "ci_upper": np.array([0.3]),
        },
        summarize_interval_coverage=lambda point, lower, upper, truth: {
            "covers": np.array([True]),
            "widths": upper - lower,
        },
    )

    outputs = runner.run_profile(
        profile_name="smoke",
        results_dir=tmp_path,
        kan_steps=100,
        kan_config_label="kan_width64_v1",
        simulation_module=fake_simulation,
        inference_module=fake_inference,
    )

    manifest = json.loads(outputs["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["kan_steps"] == 25
    assert manifest["kan_config_label"] == "kan_width64_v1"
    assert manifest["kan_config"] == {"steps": 25, "hidden_dim": 64}
    assert manifest["kan_config_id"] == "kan_hd64_st25"
    assert seen_kan_configs == [{"steps": 25, "hidden_dim": 64}, {"steps": 25, "hidden_dim": 64}]
