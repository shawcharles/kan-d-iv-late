import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from dlate_score import compute_dlate_score_objects
from kan_utils import (
    DEFAULT_PROBABILITY_EPSILON,
    GRID_SIZE,
    KAN_HIDDEN_DIM,
    KAN_LR,
    KAN_REG_STRENGTH,
    KAN_WEIGHT_DECAY,
    MIN_CLASS_COUNT_THRESHOLD,
    SPLINE_ORDER,
    build_kan_config,
    build_kan_config_id,
    build_rf_config,
    build_rf_config_id,
    clip_probabilities,
    fit_binary_kan_predict,
    fit_binary_rf_predict,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "simulation_runs"

SIMULATION_K_FOLDS = 3
SIMULATION_KAN_STEPS = 25
RF_N_ESTIMATORS = 100
PROBABILITY_EPSILON = 1e-6
DEFAULT_DATA_SEED = 20260420
DEFAULT_TRUTH_SAMPLE_SIZE = 50000
DEFAULT_TRUTH_SEED = 1729
FRAGILE_DENOMINATOR_THRESHOLD = 0.02
WEAK_FIRST_STAGE_THRESHOLD = 0.05
RF_CONFIG_ID = "rf_core_v1"
KAN_CONFIG_ID = "kan_core_v1"

STAGE_NAMES = ("pi", "p", "mu", "mu_w1", "mu_w0")
_IS_ACTUALLY_IN_COLAB = "google.colab" in sys.modules

DESIGN_LIBRARY = {
    "smooth_low": {
        "name": "smooth_low",
        "n_features": 5,
        "noise_scale": 0.55,
    },
    "baseline": {
        "name": "baseline",
        "n_features": 5,
        "noise_scale": 0.75,
    },
    "complex_local": {
        "name": "complex_local",
        "n_features": 5,
        "noise_scale": 0.95,
    },
}

STRENGTH_LIBRARY = {
    "weak": {
        "name": "weak",
        "complier_logit_shift": -1.0,
    },
    "medium": {
        "name": "medium",
        "complier_logit_shift": 0.0,
    },
    "strong": {
        "name": "strong",
        "complier_logit_shift": 0.9,
    },
}


def is_running_in_colab():
    """Return whether the script is running in Google Colab."""
    return _IS_ACTUALLY_IN_COLAB


def _logistic(values):
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _sample_categorical_rows(probabilities, rng):
    uniforms = rng.random(probabilities.shape[0])
    cumulative = np.cumsum(probabilities, axis=1)
    return (uniforms[:, None] > cumulative).sum(axis=1)


def _sanitize_label(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-")


def build_scenario_label(*, design_name, instrument_strength, n_samples, n_simulations):
    return (
        f"{_sanitize_label(design_name)}__"
        f"{_sanitize_label(instrument_strength)}__"
        f"n{n_samples}__"
        f"nsim{n_simulations}"
    )


def list_design_names():
    return list(DESIGN_LIBRARY)


def list_instrument_strength_names():
    return list(STRENGTH_LIBRARY)


def build_core_config_metadata(*, kan_steps):
    return build_model_config_metadata(kan_config={"steps": kan_steps})


def build_model_config_metadata(*, kan_config=None, rf_config=None):
    normalized_kan_config = build_kan_config(**(kan_config or {}))
    normalized_rf_config = build_rf_config(**(rf_config or {}))
    return {
        "rf_config_id": build_rf_config_id(normalized_rf_config, default_id=RF_CONFIG_ID),
        "kan_config_id": build_kan_config_id(normalized_kan_config, default_id=KAN_CONFIG_ID),
        "rf_config": normalized_rf_config,
        "kan_config": normalized_kan_config,
    }


def get_design_config(design_name):
    if design_name not in DESIGN_LIBRARY:
        raise ValueError(f"Unsupported design_name: {design_name}")
    return dict(DESIGN_LIBRARY[design_name])


def get_instrument_strength_config(instrument_strength):
    if instrument_strength not in STRENGTH_LIBRARY:
        raise ValueError(f"Unsupported instrument_strength: {instrument_strength}")
    return dict(STRENGTH_LIBRARY[instrument_strength])


def summarize_binary_training_labels(train_labels, *, min_class_count=None):
    """Summarize whether a binary learner will use a fallback prediction."""
    train_labels = np.asarray(train_labels).reshape(-1)
    if train_labels.size == 0:
        return {"used_fallback": True, "reason": "empty"}

    unique_labels, counts = np.unique(train_labels, return_counts=True)
    if len(unique_labels) == 1:
        return {"used_fallback": True, "reason": "constant"}

    if min_class_count is not None and np.min(counts) < min_class_count:
        return {"used_fallback": True, "reason": "imbalanced"}

    return {"used_fallback": False, "reason": "fit"}


def _initialize_fit_diagnostics():
    diagnostics = {}
    for stage_name in STAGE_NAMES:
        diagnostics[f"{stage_name}_calls"] = 0
        diagnostics[f"{stage_name}_fallbacks"] = 0
        diagnostics[f"{stage_name}_empty_count"] = 0
        diagnostics[f"{stage_name}_constant_count"] = 0
        diagnostics[f"{stage_name}_imbalanced_count"] = 0
    return diagnostics


def _record_fit_event(diagnostics, stage_name, label_diagnostics):
    diagnostics[f"{stage_name}_calls"] += 1
    if label_diagnostics["used_fallback"]:
        diagnostics[f"{stage_name}_fallbacks"] += 1
        diagnostics[f"{stage_name}_{label_diagnostics['reason']}_count"] += 1


def _finalize_fit_diagnostics(diagnostics, pi_hat, p_hat_0, p_hat_1):
    for stage_name in STAGE_NAMES:
        calls = diagnostics[f"{stage_name}_calls"]
        fallbacks = diagnostics[f"{stage_name}_fallbacks"]
        diagnostics[f"{stage_name}_fallback_rate"] = 0.0 if calls == 0 else fallbacks / calls

    diagnostics["pi_clip_rate"] = float(
        np.mean((pi_hat <= PROBABILITY_EPSILON) | (pi_hat >= 1 - PROBABILITY_EPSILON))
    )
    p_stack = np.column_stack([p_hat_0, p_hat_1])
    diagnostics["p_clip_rate"] = float(
        np.mean((p_stack <= PROBABILITY_EPSILON) | (p_stack >= 1 - PROBABILITY_EPSILON))
    )
    return diagnostics


def _build_instrument_probabilities(x_values, design_name):
    if design_name == "smooth_low":
        logits = 0.10 + 0.40 * x_values[:, 0] - 0.20 * x_values[:, 1] + 0.10 * x_values[:, 2]
    elif design_name == "complex_local":
        logits = (
            -0.05
            + 0.45 * np.sin(np.pi * x_values[:, 0] * x_values[:, 1])
            + 0.30 * np.exp(-(x_values[:, 2] ** 2))
            - 0.20 * x_values[:, 3]
            + 0.15 * x_values[:, 4] ** 2
        )
    else:
        logits = (
            0.2
            + 0.55 * x_values[:, 0]
            - 0.35 * x_values[:, 1]
            + 0.25 * np.sin(np.pi * x_values[:, 2])
            + 0.15 * x_values[:, 3] * x_values[:, 4]
        )
    return clip_probabilities(_logistic(logits), epsilon=PROBABILITY_EPSILON)


def _build_strata_probabilities(x_values, design_name, instrument_strength):
    if design_name == "smooth_low":
        logits = np.column_stack(
            [
                -0.10 - 0.30 * x_values[:, 0] + 0.10 * x_values[:, 1],
                0.35 + 0.20 * x_values[:, 0] + 0.20 * x_values[:, 1] - 0.10 * x_values[:, 2],
                -0.15 + 0.15 * x_values[:, 0] + 0.05 * x_values[:, 3],
            ]
        )
    elif design_name == "complex_local":
        logits = np.column_stack(
            [
                -0.20 - 0.25 * x_values[:, 0] + 0.25 * np.tanh(x_values[:, 1]) + 0.15 * x_values[:, 2],
                0.25
                + 0.55 * np.sin(np.pi * x_values[:, 0] * x_values[:, 1])
                + 0.30 * np.exp(-(x_values[:, 2] ** 2))
                - 0.15 * x_values[:, 3] * x_values[:, 4],
                -0.20 + 0.15 * x_values[:, 0] - 0.20 * np.cos(np.pi * x_values[:, 3]) + 0.10 * x_values[:, 4],
            ]
        )
    else:
        logits = np.column_stack(
            [
                -0.25 - 0.45 * x_values[:, 0] + 0.20 * x_values[:, 1] - 0.10 * x_values[:, 2],
                0.65 + 0.45 * x_values[:, 0] + 0.45 * np.sin(np.pi * x_values[:, 1]) - 0.15 * x_values[:, 2],
                -0.55 + 0.35 * x_values[:, 0] + 0.20 * x_values[:, 3],
            ]
        )

    strength_config = get_instrument_strength_config(instrument_strength)
    complier_shift = strength_config["complier_logit_shift"]
    logits = logits + np.array([-0.5 * complier_shift, complier_shift, -0.5 * complier_shift])
    return _softmax(logits)


def _build_potential_outcomes(x_values, rng, noise_scale, design_name):
    base_noise = rng.normal(scale=noise_scale, size=x_values.shape[0])
    if design_name == "smooth_low":
        y0 = 0.60 * x_values[:, 0] - 0.25 * x_values[:, 1] + 0.20 * x_values[:, 2] + base_noise
        treatment_effect = 0.55 + 0.15 * x_values[:, 0] - 0.10 * x_values[:, 1] + 0.08 * np.sin(np.pi * x_values[:, 2])
    elif design_name == "complex_local":
        y0 = (
            0.70 * np.sin(np.pi * x_values[:, 0] * x_values[:, 1])
            + 0.50 * np.exp(-((x_values[:, 2] + 0.5) ** 2))
            - 0.35 * (x_values[:, 3] > 0).astype(float)
            + 0.25 * x_values[:, 4] ** 2
            + base_noise
        )
        treatment_effect = (
            0.75
            + 0.45 * np.cos(np.pi * x_values[:, 2])
            + 0.30 * (np.abs(x_values[:, 0]) < 0.6).astype(float) * x_values[:, 1]
            - 0.15 * x_values[:, 4]
            + 0.20 * np.sin(np.pi * x_values[:, 3] * x_values[:, 4])
        )
    else:
        y0 = (
            0.80 * x_values[:, 0]
            + np.sin(np.pi * x_values[:, 1])
            + 0.35 * x_values[:, 2] ** 2
            - 0.25 * x_values[:, 3] * x_values[:, 4]
            + base_noise
        )
        treatment_effect = (
            0.90
            + 0.35 * np.cos(np.pi * x_values[:, 2])
            + 0.20 * x_values[:, 0] * x_values[:, 1]
            - 0.10 * x_values[:, 4]
        )
    y1 = y0 + treatment_effect
    return y0, y1


def generate_dlate_data(
    n_samples=2000,
    n_features=5,
    seed=42,
    design_name="baseline",
    instrument_strength="medium",
):
    """
    Generate observed data and latent principal-strata objects for D-IV-LATE simulation.

    The DGP is monotone by construction:
    - never-takers: W(0)=0, W(1)=0
    - compliers:    W(0)=0, W(1)=1
    - always-takers:W(0)=1, W(1)=1
    """
    design = get_design_config(design_name)
    if n_features != design["n_features"]:
        raise ValueError(
            f"Design '{design_name}' expects {design['n_features']} features, got {n_features}"
        )

    rng = np.random.default_rng(seed)
    x_values = rng.normal(size=(n_samples, n_features))
    z_values = rng.binomial(1, _build_instrument_probabilities(x_values, design_name))

    strata_codes = _sample_categorical_rows(
        _build_strata_probabilities(x_values, design_name, instrument_strength),
        rng,
    )
    strata_labels = np.array(["never", "complier", "always"])[strata_codes]

    w0 = (strata_labels == "always").astype(int)
    w1 = ((strata_labels == "always") | (strata_labels == "complier")).astype(int)
    w_values = np.where(z_values == 1, w1, w0)

    y0, y1 = _build_potential_outcomes(x_values, rng, design["noise_scale"], design_name)
    y_values = np.where(w_values == 1, y1, y0)

    data = pd.DataFrame(x_values, columns=[f"X{i}" for i in range(n_features)])
    data["Z"] = z_values
    data["W"] = w_values
    data["Y"] = y_values

    latent = {
        "Y0": y0,
        "Y1": y1,
        "W0": w0,
        "W1": w1,
        "strata": strata_labels,
        "complier_share": float(np.mean(strata_labels == "complier")),
        "true_first_stage": float(np.mean(w1 - w0)),
        "design_name": design_name,
        "instrument_strength": instrument_strength,
    }
    return data, latent


def build_truth_bundle(
    *,
    design_name="baseline",
    instrument_strength="medium",
    y_points=10,
    truth_sample_size=DEFAULT_TRUTH_SAMPLE_SIZE,
    truth_seed=DEFAULT_TRUTH_SEED,
):
    """
    Build a deterministic design-level truth bundle for the complier D-IV-LATE.

    The reported truth is computed from the complier potential-outcome
    distributions in a large reference population, not from the realized
    estimation sample.
    """
    truth_data, truth_latent = generate_dlate_data(
        n_samples=truth_sample_size,
        seed=truth_seed,
        design_name=design_name,
        instrument_strength=instrument_strength,
    )
    complier_mask = truth_latent["strata"] == "complier"
    if not np.any(complier_mask):
        raise ValueError("Truth bundle contains no compliers; adjust the DGP.")

    y_grid = np.quantile(truth_data["Y"], np.linspace(0.01, 0.99, y_points))
    cdf_y1 = np.array([np.mean(truth_latent["Y1"][complier_mask] <= y_val) for y_val in y_grid])
    cdf_y0 = np.array([np.mean(truth_latent["Y0"][complier_mask] <= y_val) for y_val in y_grid])
    true_dlate = cdf_y1 - cdf_y0

    truth_df = pd.DataFrame(
        {
            "y": y_grid,
            "true_dlate": true_dlate,
            "cdf_y1_complier": cdf_y1,
            "cdf_y0_complier": cdf_y0,
        }
    )
    truth_meta = {
        "design_name": design_name,
        "instrument_strength": instrument_strength,
        "truth_sample_size": truth_sample_size,
        "truth_seed": truth_seed,
        "true_complier_share": float(truth_latent["complier_share"]),
        "true_first_stage": float(truth_latent["true_first_stage"]),
    }
    return {"truth_df": truth_df, "meta": truth_meta}


def _predict_binary(
    model_type,
    train_features,
    train_labels,
    test_features,
    *,
    kan_config,
    rf_config,
    clip=False,
):
    if model_type == "kan":
        label_diagnostics = summarize_binary_training_labels(
            train_labels,
            min_class_count=MIN_CLASS_COUNT_THRESHOLD,
        )
        kan_predict_kwargs = dict(kan_config)
        kan_predict_kwargs["epsilon"] = kan_predict_kwargs.pop("probability_epsilon")
        predictions = fit_binary_kan_predict(
            train_features,
            train_labels,
            test_features,
            **kan_predict_kwargs,
            clip=clip,
        )
    elif model_type == "rf":
        label_diagnostics = summarize_binary_training_labels(train_labels)
        predictions = fit_binary_rf_predict(
            train_features,
            train_labels,
            test_features,
            **rf_config,
            clip=clip,
            epsilon=PROBABILITY_EPSILON,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return predictions, label_diagnostics


def estimate_nuisance_functions(
    data,
    y_grid,
    *,
    model_type="kan",
    k_folds=SIMULATION_K_FOLDS,
    kan_steps=SIMULATION_KAN_STEPS,
    kan_config=None,
    rf_config=None,
):
    """Estimate the reduced-form nuisance bundle and expose fit diagnostics."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    kan_config = dict(kan_config or {})
    kan_config.setdefault("steps", kan_steps)
    kan_config = build_kan_config(**kan_config)
    rf_config = build_rf_config(**(rf_config or {}))

    x_cols = [col for col in data.columns if col.startswith("X")]
    n_obs = len(data)
    n_grid = len(y_grid)

    pi_hat = np.zeros(n_obs)
    p_hat_0 = np.zeros(n_obs)
    p_hat_1 = np.zeros(n_obs)
    mu_hat_0 = np.zeros((n_obs, n_grid))
    mu_hat_1 = np.zeros((n_obs, n_grid))
    fit_diagnostics = _initialize_fit_diagnostics()

    for train_index, test_index in kf.split(data):
        data_train = data.iloc[train_index]
        data_test = data.iloc[test_index]

        x_test = data_test[x_cols].to_numpy()
        xz_test_z0 = np.column_stack([x_test, np.zeros(len(data_test), dtype=float)])
        xz_test_z1 = np.column_stack([x_test, np.ones(len(data_test), dtype=float)])

        pi_predictions, pi_meta = _predict_binary(
            model_type,
            data_train[x_cols].to_numpy(),
            data_train["Z"].to_numpy(),
            x_test,
            kan_config=kan_config,
            rf_config=rf_config,
            clip=True,
        )
        _record_fit_event(fit_diagnostics, "pi", pi_meta)
        pi_hat[test_index] = pi_predictions

        p_predictions_z0, p_meta = _predict_binary(
            model_type,
            data_train[x_cols + ["Z"]].to_numpy(),
            data_train["W"].to_numpy(),
            xz_test_z0,
            kan_config=kan_config,
            rf_config=rf_config,
            clip=True,
        )
        p_predictions_z1, _ = _predict_binary(
            model_type,
            data_train[x_cols + ["Z"]].to_numpy(),
            data_train["W"].to_numpy(),
            xz_test_z1,
            kan_config=kan_config,
            rf_config=rf_config,
            clip=True,
        )
        _record_fit_event(fit_diagnostics, "p", p_meta)
        p_hat_0[test_index] = p_predictions_z0
        p_hat_1[test_index] = p_predictions_z1

        for y_idx, y_val in enumerate(y_grid):
            y_indicator = (data_train["Y"] <= y_val).astype(int).to_numpy()
            mu_predictions_z0, mu_meta = _predict_binary(
                model_type,
                data_train[x_cols + ["Z"]].to_numpy(),
                y_indicator,
                xz_test_z0,
                kan_config=kan_config,
                rf_config=rf_config,
            )
            mu_predictions_z1, _ = _predict_binary(
                model_type,
                data_train[x_cols + ["Z"]].to_numpy(),
                y_indicator,
                xz_test_z1,
                kan_config=kan_config,
                rf_config=rf_config,
            )
            _record_fit_event(fit_diagnostics, "mu", mu_meta)
            _record_fit_event(fit_diagnostics, "mu_w0", mu_meta)
            _record_fit_event(fit_diagnostics, "mu_w1", mu_meta)
            mu_hat_0[test_index, y_idx] = mu_predictions_z0
            mu_hat_1[test_index, y_idx] = mu_predictions_z1

    fit_diagnostics = _finalize_fit_diagnostics(fit_diagnostics, pi_hat, p_hat_0, p_hat_1)
    return {
        "pi_hat": pi_hat,
        "p_hat_0": p_hat_0,
        "p_hat_1": p_hat_1,
        "mu_hat_0": mu_hat_0,
        "mu_hat_1": mu_hat_1,
        "fit_diagnostics": fit_diagnostics,
    }


def dlate_estimator(data, nuisance_results, y_grid, *, return_diagnostics=False):
    """Estimate the D-LATE curve from the reduced-form nuisance bundle."""
    score_objects = compute_dlate_score_objects(
        data,
        nuisance_results,
        y_grid,
        epsilon=PROBABILITY_EPSILON,
    )
    estimator_diagnostics = {
        "mean_psi_beta": score_objects["mean_psi_beta"],
        "abs_mean_psi_beta": score_objects["abs_mean_psi_beta"],
        "near_zero_denominator": score_objects["near_zero_denominator"],
    }
    if return_diagnostics:
        return score_objects["dlate"], estimator_diagnostics
    return score_objects["dlate"]


def compute_sample_first_stage(data):
    """Compute the sample first-stage difference E[W|Z=1] - E[W|Z=0]."""
    grouped = data.groupby("Z")["W"].mean()
    if 0 not in grouped.index or 1 not in grouped.index:
        return np.nan
    return float(grouped.loc[1] - grouped.loc[0])


def build_run_label(
    *,
    design_name,
    instrument_strength,
    n_simulations,
    n_samples,
    y_points,
    seed_base,
    tag=None,
):
    parts = [
        _sanitize_label(design_name),
        _sanitize_label(instrument_strength),
        f"nsim{n_simulations}",
        f"n{n_samples}",
        f"y{y_points}",
        f"seed{seed_base}",
    ]
    if tag:
        parts.append(_sanitize_label(tag))
    return "__".join(parts)


def run_simulation(
    *,
    n_simulations=50,
    n_samples=2000,
    model_type="kan",
    design_name="baseline",
    instrument_strength="medium",
    y_points=10,
    k_folds=SIMULATION_K_FOLDS,
    kan_steps=SIMULATION_KAN_STEPS,
    kan_config=None,
    rf_config=None,
    seed_base=DEFAULT_DATA_SEED,
    truth_bundle=None,
    truth_seed=DEFAULT_TRUTH_SEED,
    profile_name="standalone",
    scenario_label=None,
    run_label=None,
    rf_config_id=RF_CONFIG_ID,
    kan_config_id=KAN_CONFIG_ID,
):
    """Run the benchmark loop for one model family against a fixed truth bundle."""
    kan_config = dict(kan_config or {})
    kan_config.setdefault("steps", kan_steps)
    kan_config = build_kan_config(**kan_config)
    rf_config = build_rf_config(**(rf_config or {}))
    if truth_bundle is None:
        truth_bundle = build_truth_bundle(
            design_name=design_name,
            instrument_strength=instrument_strength,
            y_points=y_points,
        )

    truth_df = truth_bundle["truth_df"]
    y_grid = truth_df["y"].to_numpy()
    true_dlate = truth_df["true_dlate"].to_numpy()
    truth_sample_size = truth_bundle["meta"]["truth_sample_size"]
    scenario_label = scenario_label or build_scenario_label(
        design_name=design_name,
        instrument_strength=instrument_strength,
        n_samples=n_samples,
        n_simulations=n_simulations,
    )
    run_label = run_label or build_run_label(
        design_name=design_name,
        instrument_strength=instrument_strength,
        n_simulations=n_simulations,
        n_samples=n_samples,
        y_points=y_points,
        seed_base=seed_base,
        tag=profile_name,
    )
    model_config_id = kan_config_id if model_type == "kan" else rf_config_id

    result_rows = []
    diagnostic_rows = []

    for simulation_idx in range(n_simulations):
        print(f"Running simulation {simulation_idx + 1}/{n_simulations} for {model_type}...")
        simulation_seed = seed_base + simulation_idx
        runtime_start = time.perf_counter()
        data, latent = generate_dlate_data(
            n_samples=n_samples,
            seed=simulation_seed,
            design_name=design_name,
            instrument_strength=instrument_strength,
        )
        nuisance_results = estimate_nuisance_functions(
            data,
            y_grid,
            model_type=model_type,
            k_folds=k_folds,
            kan_steps=kan_steps,
            kan_config=kan_config,
            rf_config=rf_config,
        )
        dlate_est, estimator_diagnostics = dlate_estimator(
            data,
            nuisance_results,
            y_grid,
            return_diagnostics=True,
        )
        runtime_sec = float(time.perf_counter() - runtime_start)

        sample_first_stage = compute_sample_first_stage(data)
        fit_diagnostics = nuisance_results["fit_diagnostics"]
        weak_first_stage = int(np.abs(sample_first_stage) < WEAK_FIRST_STAGE_THRESHOLD)
        fragile_denominator = int(
            estimator_diagnostics["abs_mean_psi_beta"] < FRAGILE_DENOMINATOR_THRESHOLD
        )
        shared_metadata = {
            "profile_name": profile_name,
            "scenario_label": scenario_label,
            "run_label": run_label,
            "design_name": design_name,
            "instrument_strength": instrument_strength,
            "n_samples": n_samples,
            "n_simulations": n_simulations,
            "y_points": y_points,
            "k_folds": k_folds,
            "kan_steps": kan_config["steps"],
            "kan_hidden_dim": kan_config["hidden_dim"],
            "kan_grid_size": kan_config["grid_size"],
            "kan_spline_order": kan_config["spline_order"],
            "kan_lr": kan_config["lr"],
            "kan_weight_decay": kan_config["weight_decay"],
            "kan_reg_strength": kan_config["reg_strength"],
            "kan_min_class_count": kan_config["min_class_count"],
            "probability_epsilon": kan_config["probability_epsilon"],
            "truth_sample_size": truth_sample_size,
            "truth_seed": truth_seed,
            "seed_base": seed_base,
            "rf_config_id": rf_config_id,
            "kan_config_id": kan_config_id,
            "model": model_type,
            "model_config_id": model_config_id,
        }
        diagnostic_row = {
            **shared_metadata,
            "simulation": simulation_idx,
            "seed": simulation_seed,
            "sample_first_stage": sample_first_stage,
            "true_complier_share": float(latent["complier_share"]),
            "true_first_stage": float(latent["true_first_stage"]),
            "mean_psi_beta": estimator_diagnostics["mean_psi_beta"],
            "abs_mean_psi_beta": estimator_diagnostics["abs_mean_psi_beta"],
            "near_zero_denominator": int(estimator_diagnostics["near_zero_denominator"]),
            "fragile_denominator": fragile_denominator,
            "weak_first_stage": weak_first_stage,
            "runtime_sec": runtime_sec,
        }
        diagnostic_row.update(fit_diagnostics)
        diagnostic_rows.append(diagnostic_row)

        for y_idx, y_val in enumerate(y_grid):
            result_rows.append(
                {
                    **shared_metadata,
                    "simulation": simulation_idx,
                    "seed": simulation_seed,
                    "y": y_val,
                    "dlate_est": dlate_est[y_idx],
                    "true_dlate": true_dlate[y_idx],
                    "bias": dlate_est[y_idx] - true_dlate[y_idx],
                    "abs_error": np.abs(dlate_est[y_idx] - true_dlate[y_idx]),
                    "sample_first_stage": sample_first_stage,
                    "true_complier_share": float(latent["complier_share"]),
                    "true_first_stage": float(latent["true_first_stage"]),
                    "mean_psi_beta": estimator_diagnostics["mean_psi_beta"],
                    "abs_mean_psi_beta": estimator_diagnostics["abs_mean_psi_beta"],
                    "near_zero_denominator": int(estimator_diagnostics["near_zero_denominator"]),
                    "fragile_denominator": fragile_denominator,
                    "weak_first_stage": weak_first_stage,
                    "runtime_sec": runtime_sec,
                }
            )

    return pd.DataFrame(result_rows), pd.DataFrame(diagnostic_rows)


def summarize_benchmark_outputs(replication_results, diagnostic_results):
    """Build summary tables from replication-level simulation outputs."""
    pointwise_group_columns = [
        "profile_name",
        "scenario_label",
        "run_label",
        "design_name",
        "instrument_strength",
        "n_samples",
        "n_simulations",
        "y_points",
        "k_folds",
        "kan_steps",
        "kan_hidden_dim",
        "kan_grid_size",
        "kan_spline_order",
        "kan_lr",
        "kan_weight_decay",
        "kan_reg_strength",
        "kan_min_class_count",
        "probability_epsilon",
        "truth_sample_size",
        "truth_seed",
        "seed_base",
        "rf_config_id",
        "kan_config_id",
        "model",
        "y",
        "true_dlate",
    ]
    summary_df = (
        replication_results.groupby(pointwise_group_columns, as_index=False)
        .agg(
            mean_estimate=("dlate_est", "mean"),
            bias=("bias", "mean"),
            rmse=("bias", lambda values: float(np.sqrt(np.nanmean(np.square(values))))),
            mae=("abs_error", "mean"),
            nan_rate=("dlate_est", lambda values: float(np.mean(pd.isna(values)))),
        )
    )
    summary_df["pointwise_bias"] = summary_df["bias"]
    summary_df["pointwise_rmse"] = summary_df["rmse"]
    summary_df["pointwise_mae"] = summary_df["mae"]
    summary_df["pointwise_nan_rate"] = summary_df["nan_rate"]

    scenario_group_columns = [
        "profile_name",
        "scenario_label",
        "run_label",
        "design_name",
        "instrument_strength",
        "n_samples",
        "n_simulations",
        "y_points",
        "k_folds",
        "kan_steps",
        "kan_hidden_dim",
        "kan_grid_size",
        "kan_spline_order",
        "kan_lr",
        "kan_weight_decay",
        "kan_reg_strength",
        "kan_min_class_count",
        "probability_epsilon",
        "truth_sample_size",
        "truth_seed",
        "seed_base",
        "rf_config_id",
        "kan_config_id",
        "model",
        "model_config_id",
    ]
    error_summary = (
        replication_results.groupby(scenario_group_columns, as_index=False)
        .agg(
            scenario_signed_bias=("bias", "mean"),
            scenario_mean_abs_error=("abs_error", "mean"),
            scenario_integrated_rmse=("bias", lambda values: float(np.sqrt(np.nanmean(np.square(values))))),
            scenario_nan_rate=("dlate_est", lambda values: float(np.mean(pd.isna(values)))),
        )
    )

    diagnostic_summary = (
        diagnostic_results.groupby(scenario_group_columns, as_index=False)
        .agg(
            replications=("simulation", "nunique"),
            mean_sample_first_stage=("sample_first_stage", "mean"),
            min_sample_first_stage=("sample_first_stage", "min"),
            weak_first_stage_share=("weak_first_stage", "mean"),
            mean_true_complier_share=("true_complier_share", "mean"),
            mean_true_first_stage=("true_first_stage", "mean"),
            mean_abs_psi_beta=("abs_mean_psi_beta", "mean"),
            near_zero_denominator_share=("near_zero_denominator", "mean"),
            fragile_denominator_share=("fragile_denominator", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
            total_runtime_sec=("runtime_sec", "sum"),
            mean_pi_clip_rate=("pi_clip_rate", "mean"),
            mean_p_clip_rate=("p_clip_rate", "mean"),
            mean_pi_fallback_rate=("pi_fallback_rate", "mean"),
            mean_p_fallback_rate=("p_fallback_rate", "mean"),
            mean_mu_fallback_rate=("mu_fallback_rate", "mean"),
            mean_mu_w1_fallback_rate=("mu_w1_fallback_rate", "mean"),
            mean_mu_w0_fallback_rate=("mu_w0_fallback_rate", "mean"),
        )
    )

    scenario_summary = error_summary.merge(
        diagnostic_summary,
        on=scenario_group_columns,
        how="outer",
        validate="one_to_one",
    )

    return summary_df, scenario_summary


def build_compatibility_summary(summary_df):
    """Build the legacy wide summary CSV used by the smoke test and prior workflow."""
    compatibility = pd.DataFrame({"y": np.sort(summary_df["y"].unique())})

    for model_name in ("kan", "rf"):
        model_summary = summary_df[summary_df["model"] == model_name][["y", "bias", "rmse"]].rename(
            columns={
                "bias": f"{model_name}_avg_bias",
                "rmse": f"{model_name}_rmse",
            }
        )
        compatibility = compatibility.merge(model_summary, on="y", how="left")

    return compatibility


def write_benchmark_outputs(
    *,
    results_dir,
    run_label,
    truth_bundle,
    replication_results,
    summary_df,
    scenario_summary,
    diagnostic_results,
    manifest,
):
    """Write simulation outputs and return the output paths."""
    results_dir.mkdir(parents=True, exist_ok=True)

    truth_path = results_dir / f"simulation_truth_{run_label}.csv"
    summary_path = results_dir / f"simulation_summary_{run_label}.csv"
    replications_path = results_dir / f"simulation_replications_{run_label}.csv"
    diagnostics_path = results_dir / f"simulation_diagnostics_{run_label}.csv"
    diagnostics_summary_path = results_dir / f"simulation_diagnostics_summary_{run_label}.csv"
    manifest_path = results_dir / f"simulation_manifest_{run_label}.json"
    compatibility_path = results_dir / "simulation_results.csv"

    truth_bundle["truth_df"].to_csv(truth_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    replication_results.to_csv(replications_path, index=False)
    diagnostic_results.to_csv(diagnostics_path, index=False)
    scenario_summary.to_csv(diagnostics_summary_path, index=False)
    build_compatibility_summary(summary_df).to_csv(compatibility_path, index=False)

    manifest = dict(manifest)
    manifest["output_files"] = {
        "truth": str(truth_path),
        "summary": str(summary_path),
        "replications": str(replications_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_summary": str(diagnostics_summary_path),
        "scenario_summary": str(diagnostics_summary_path),
        "compatibility_summary": str(compatibility_path),
        "manifest": str(manifest_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return {
        "truth": truth_path,
        "summary": summary_path,
        "replications": replications_path,
        "diagnostics": diagnostics_path,
        "diagnostics_summary": diagnostics_summary_path,
        "manifest": manifest_path,
        "compatibility_summary": compatibility_path,
    }


def main(
    *,
    results_dir=DEFAULT_RESULTS_DIR,
    n_simulations=50,
    n_samples=2000,
    design_name="baseline",
    instrument_strength="medium",
    y_points=10,
    k_folds=SIMULATION_K_FOLDS,
    kan_steps=SIMULATION_KAN_STEPS,
    kan_hidden_dim=KAN_HIDDEN_DIM,
    kan_grid_size=GRID_SIZE,
    kan_spline_order=SPLINE_ORDER,
    kan_lr=KAN_LR,
    kan_weight_decay=KAN_WEIGHT_DECAY,
    kan_reg_strength=KAN_REG_STRENGTH,
    kan_min_class_count=MIN_CLASS_COUNT_THRESHOLD,
    probability_epsilon=DEFAULT_PROBABILITY_EPSILON,
    truth_sample_size=DEFAULT_TRUTH_SAMPLE_SIZE,
    truth_seed=DEFAULT_TRUTH_SEED,
    seed_base=DEFAULT_DATA_SEED,
    tag=None,
    profile_name="standalone",
    scenario_label=None,
):
    print("Starting simulation study...")
    results_dir = Path(results_dir)
    scenario_label = scenario_label or build_scenario_label(
        design_name=design_name,
        instrument_strength=instrument_strength,
        n_samples=n_samples,
        n_simulations=n_simulations,
    )
    run_label = build_run_label(
        design_name=design_name,
        instrument_strength=instrument_strength,
        n_simulations=n_simulations,
        n_samples=n_samples,
        y_points=y_points,
        seed_base=seed_base,
        tag=tag,
    )
    core_config_metadata = build_model_config_metadata(
        kan_config={
            "steps": kan_steps,
            "hidden_dim": kan_hidden_dim,
            "grid_size": kan_grid_size,
            "spline_order": kan_spline_order,
            "lr": kan_lr,
            "weight_decay": kan_weight_decay,
            "reg_strength": kan_reg_strength,
            "min_class_count": kan_min_class_count,
            "probability_epsilon": probability_epsilon,
        }
    )
    truth_bundle = build_truth_bundle(
        design_name=design_name,
        instrument_strength=instrument_strength,
        y_points=y_points,
        truth_sample_size=truth_sample_size,
        truth_seed=truth_seed,
    )

    kan_results, kan_diagnostics = run_simulation(
        n_simulations=n_simulations,
        n_samples=n_samples,
        model_type="kan",
        design_name=design_name,
        instrument_strength=instrument_strength,
        y_points=y_points,
        k_folds=k_folds,
        kan_steps=kan_steps,
        kan_config=core_config_metadata["kan_config"],
        rf_config=core_config_metadata["rf_config"],
        seed_base=seed_base,
        truth_bundle=truth_bundle,
        truth_seed=truth_seed,
        profile_name=profile_name,
        scenario_label=scenario_label,
        run_label=run_label,
        rf_config_id=core_config_metadata["rf_config_id"],
        kan_config_id=core_config_metadata["kan_config_id"],
    )
    rf_results, rf_diagnostics = run_simulation(
        n_simulations=n_simulations,
        n_samples=n_samples,
        model_type="rf",
        design_name=design_name,
        instrument_strength=instrument_strength,
        y_points=y_points,
        k_folds=k_folds,
        kan_steps=kan_steps,
        kan_config=core_config_metadata["kan_config"],
        rf_config=core_config_metadata["rf_config"],
        seed_base=seed_base,
        truth_bundle=truth_bundle,
        truth_seed=truth_seed,
        profile_name=profile_name,
        scenario_label=scenario_label,
        run_label=run_label,
        rf_config_id=core_config_metadata["rf_config_id"],
        kan_config_id=core_config_metadata["kan_config_id"],
    )

    replication_results = pd.concat([kan_results, rf_results], ignore_index=True)
    diagnostic_results = pd.concat([kan_diagnostics, rf_diagnostics], ignore_index=True)
    summary_df, scenario_summary = summarize_benchmark_outputs(replication_results, diagnostic_results)
    manifest = {
        "profile_name": profile_name,
        "scenario_label": scenario_label,
        "design_name": design_name,
        "instrument_strength": instrument_strength,
        "n_simulations": n_simulations,
        "n_samples": n_samples,
        "y_points": y_points,
        "k_folds": k_folds,
        "kan_steps": core_config_metadata["kan_config"]["steps"],
        "kan_hidden_dim": core_config_metadata["kan_config"]["hidden_dim"],
        "kan_grid_size": core_config_metadata["kan_config"]["grid_size"],
        "kan_spline_order": core_config_metadata["kan_config"]["spline_order"],
        "kan_lr": core_config_metadata["kan_config"]["lr"],
        "kan_weight_decay": core_config_metadata["kan_config"]["weight_decay"],
        "kan_reg_strength": core_config_metadata["kan_config"]["reg_strength"],
        "kan_min_class_count": core_config_metadata["kan_config"]["min_class_count"],
        "probability_epsilon": core_config_metadata["kan_config"]["probability_epsilon"],
        "truth_sample_size": truth_sample_size,
        "truth_seed": truth_seed,
        "seed_base": seed_base,
        "run_label": run_label,
        "rf_config_id": core_config_metadata["rf_config_id"],
        "kan_config_id": core_config_metadata["kan_config_id"],
        "rf_config": core_config_metadata["rf_config"],
        "kan_config": core_config_metadata["kan_config"],
        "truth_meta": truth_bundle["meta"],
    }

    output_paths = write_benchmark_outputs(
        results_dir=results_dir,
        run_label=run_label,
        truth_bundle=truth_bundle,
        replication_results=replication_results,
        summary_df=summary_df,
        scenario_summary=scenario_summary,
        diagnostic_results=diagnostic_results,
        manifest=manifest,
    )

    compatibility_summary = build_compatibility_summary(summary_df)
    print("\n--- Compatibility Summary ---")
    print(compatibility_summary)
    print("\n--- Scenario Summary ---")
    print(scenario_summary)
    print(f"\nVersioned summary written to {output_paths['summary']}")
    print(f"Replication diagnostics written to {output_paths['diagnostics']}")
    print(f"Manifest written to {output_paths['manifest']}")

    if is_running_in_colab():
        zip_and_download_results_colab([str(path) for path in output_paths.values()])
        notif()

    print("\n--- End of Simulation ---")
    return {
        "run_label": run_label,
        "scenario_label": scenario_label,
        "profile_name": profile_name,
        "manifest": manifest,
        "output_paths": output_paths,
    }


if __name__ == "__main__":
    main()


if _IS_ACTUALLY_IN_COLAB:
    from google.colab import files, output

    def notif():
        """Play a notification sound in Colab."""
        output.eval_js(
            'new Audio("https://notificationsounds.com/message-tones/appointed-529/download/mp3").play()'
        )

    def zip_and_download_results_colab(files_to_zip, zip_filename="simulation_results.zip"):
        """Zip selected result files and download them in Colab."""
        valid_files_to_zip = [item for item in files_to_zip if os.path.exists(item)]
        if not valid_files_to_zip:
            print(f"No valid files found to zip from the list: {files_to_zip}. Download skipped.")
            return

        temp_zip_dir = "temp_zip_contents_for_download"
        final_zip_path = os.path.abspath(zip_filename)
        actual_zip_file_created = final_zip_path

        try:
            if os.path.exists(temp_zip_dir):
                shutil.rmtree(temp_zip_dir)
            os.makedirs(temp_zip_dir)

            for item_path in valid_files_to_zip:
                if os.path.isfile(item_path):
                    shutil.copy(item_path, os.path.join(temp_zip_dir, os.path.basename(item_path)))
                elif os.path.isdir(item_path):
                    destination_dir_in_zip = os.path.join(temp_zip_dir, os.path.basename(item_path))
                    shutil.copytree(
                        item_path,
                        destination_dir_in_zip,
                        ignore=shutil.ignore_patterns("sample_data"),
                    )

            archive_base_name = os.path.splitext(final_zip_path)[0]
            shutil.make_archive(archive_base_name, "zip", root_dir=os.getcwd(), base_dir=temp_zip_dir)
            actual_zip_file_created = archive_base_name + ".zip"

            if os.path.exists(actual_zip_file_created):
                print(f"Results zipped to {actual_zip_file_created}. Attempting download...")
                files.download(actual_zip_file_created)
                print(f"Download of {actual_zip_file_created} initiated.")
            else:
                print(f"Error: Zip file {actual_zip_file_created} not found after archiving.")
        except Exception as exc:  # pragma: no cover - Colab-only path
            print(f"Error during zipping or download: {exc}")
        finally:
            if os.path.exists(actual_zip_file_created):
                try:
                    os.remove(actual_zip_file_created)
                except Exception as exc:  # pragma: no cover - Colab-only path
                    print(f"Error removing zip file: {exc}")
            if os.path.exists(temp_zip_dir):
                try:
                    shutil.rmtree(temp_zip_dir)
                except Exception as exc:  # pragma: no cover - Colab-only path
                    print(f"Error removing temp directory: {exc}")
