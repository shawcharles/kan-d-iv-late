from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from dlate_score import compute_dlate_score_objects as compute_dlate_score_objects_common
from kan_utils import (
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
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "pension.csv"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"

EMPIRICAL_K_FOLDS = 5
EMPIRICAL_KAN_STEPS = 100
PROBABILITY_EPSILON = 1e-5


def load_and_prepare_data(csv_path=DEFAULT_DATA_PATH):
    """Load the pension dataset and return the active variables."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: Data file not found at {csv_path}")
        return pd.DataFrame(), []

    df = pd.read_csv(csv_path)

    y_col = "net_tfa"
    w_col = "p401"
    z_col = "e401"
    x_cols = ["inc", "age", "educ", "marr", "fsize", "twoearn", "db", "pira", "hown"]

    required_cols = [y_col, w_col, z_col] + x_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the dataset: {missing_cols}")
        return pd.DataFrame(), []

    df_subset = df[required_cols].copy()
    df_subset.dropna(inplace=True)

    if df_subset.empty:
        print("Error: No data remaining after dropping NaNs.")
        return pd.DataFrame(), []

    df_subset.rename(columns={y_col: "Y", w_col: "W", z_col: "Z"}, inplace=True)
    return df_subset, x_cols


def is_binary_column(series):
    values = set(pd.Series(series).dropna().unique().tolist())
    return values.issubset({0, 1})


def preprocess_empirical_data(data, x_cols, *, preprocess_mode="raw"):
    """Apply a simple, transparent preprocessing policy to covariates."""
    processed = data.copy()
    standardized_columns = []

    if preprocess_mode == "raw":
        return processed, {"preprocess_mode": preprocess_mode, "standardized_columns": standardized_columns}
    if preprocess_mode != "standardized":
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    for column in x_cols:
        if is_binary_column(processed[column]):
            continue
        std = float(processed[column].std(ddof=0))
        if std <= 0:
            continue
        mean = float(processed[column].mean())
        processed[column] = (processed[column] - mean) / std
        standardized_columns.append(column)

    return processed, {"preprocess_mode": preprocess_mode, "standardized_columns": standardized_columns}


def build_empirical_y_grid(data, *, y_grid_points=30, quantile_bounds=(0.01, 0.99)):
    """Build the outcome grid used by the empirical D-IV-LATE curve."""
    lower_q, upper_q = quantile_bounds
    min_y = float(np.quantile(data["Y"], lower_q))
    max_y = float(np.quantile(data["Y"], upper_q))
    if min_y >= max_y:
        min_y = float(data["Y"].min())
        max_y = float(data["Y"].max())
    if min_y >= max_y:
        raise ValueError("Outcome variable has insufficient variation for a valid y grid.")
    return np.linspace(min_y, max_y, y_grid_points)


def estimate_instrument_propensity_empirical(
    data,
    x_cols,
    *,
    model_type="rf",
    k_folds=EMPIRICAL_K_FOLDS,
    kan_steps=EMPIRICAL_KAN_STEPS,
    kan_config=None,
    rf_config=None,
):
    """Estimate out-of-fold instrument propensity scores only."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    kan_config = dict(kan_config or {})
    kan_config.setdefault("steps", kan_steps)
    kan_config = build_kan_config(**kan_config)
    rf_config = build_rf_config(**(rf_config or {}))

    pi_hat_oos = np.zeros(len(data))
    for train_index, test_index in kf.split(data):
        data_train = data.iloc[train_index]
        data_test = data.iloc[test_index]
        pi_hat_oos[test_index] = _predict_binary(
            model_type,
            data_train[x_cols].to_numpy(),
            data_train["Z"].to_numpy(),
            data_test[x_cols].to_numpy(),
            kan_config=kan_config,
            rf_config=rf_config,
            clip=True,
        )
    return clip_probabilities(pi_hat_oos, epsilon=PROBABILITY_EPSILON)


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
        kan_predict_kwargs = dict(kan_config)
        kan_predict_kwargs["epsilon"] = kan_predict_kwargs.pop("probability_epsilon")
        return fit_binary_kan_predict(
            train_features,
            train_labels,
            test_features,
            **kan_predict_kwargs,
            clip=clip,
        )
    if model_type == "rf":
        return fit_binary_rf_predict(
            train_features,
            train_labels,
            test_features,
            **rf_config,
            clip=clip,
            epsilon=PROBABILITY_EPSILON,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def estimate_nuisance_functions_empirical(
    data,
    x_cols,
    y_grid,
    *,
    model_type="kan",
    k_folds=EMPIRICAL_K_FOLDS,
    kan_steps=EMPIRICAL_KAN_STEPS,
    kan_config=None,
    rf_config=None,
):
    """
    Estimate the reduced-form nuisance bundle via cross-fitting.

    - pi_hat(X) = P(Z=1 | X)
    - p_hat_0(X) = P(W=1 | Z=0, X)
    - p_hat_1(X) = P(W=1 | Z=1, X)
    - mu_hat_0(y, X) = E[1{Y<=y} | Z=0, X]
    - mu_hat_1(y, X) = E[1{Y<=y} | Z=1, X]
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    kan_config = dict(kan_config or {})
    kan_config.setdefault("steps", kan_steps)
    kan_config = build_kan_config(**kan_config)
    rf_config = build_rf_config(**(rf_config or {}))

    n_obs = len(data)
    n_grid = len(y_grid)
    pi_hat_oos = np.zeros(n_obs)
    p_hat_0_oos = np.zeros(n_obs)
    p_hat_1_oos = np.zeros(n_obs)
    mu_hat_0_oos = np.zeros((n_obs, n_grid))
    mu_hat_1_oos = np.zeros((n_obs, n_grid))

    for train_index, test_index in kf.split(data):
        data_train = data.iloc[train_index]
        data_test = data.iloc[test_index]

        x_test = data_test[x_cols].to_numpy()
        xz_test_z0 = np.column_stack([x_test, np.zeros(len(data_test), dtype=float)])
        xz_test_z1 = np.column_stack([x_test, np.ones(len(data_test), dtype=float)])

        pi_hat_oos[test_index] = _predict_binary(
            model_type,
            data_train[x_cols].to_numpy(),
            data_train["Z"].to_numpy(),
            x_test,
            kan_config=kan_config,
            rf_config=rf_config,
            clip=True,
        )

        p_hat_0_oos[test_index] = _predict_binary(
            model_type,
            data_train[x_cols + ["Z"]].to_numpy(),
            data_train["W"].to_numpy(),
            xz_test_z0,
            kan_config=kan_config,
            rf_config=rf_config,
            clip=True,
        )
        p_hat_1_oos[test_index] = _predict_binary(
            model_type,
            data_train[x_cols + ["Z"]].to_numpy(),
            data_train["W"].to_numpy(),
            xz_test_z1,
            kan_config=kan_config,
            rf_config=rf_config,
            clip=True,
        )

        for y_idx, y_val in enumerate(y_grid):
            y_indicator = (data_train["Y"] <= y_val).astype(int).to_numpy()

            mu_hat_0_oos[test_index, y_idx] = _predict_binary(
                model_type,
                data_train[x_cols + ["Z"]].to_numpy(),
                y_indicator,
                xz_test_z0,
                kan_config=kan_config,
                rf_config=rf_config,
            )
            mu_hat_1_oos[test_index, y_idx] = _predict_binary(
                model_type,
                data_train[x_cols + ["Z"]].to_numpy(),
                y_indicator,
                xz_test_z1,
                kan_config=kan_config,
                rf_config=rf_config,
            )

    return {
        "pi_hat": pi_hat_oos,
        "p_hat_0": p_hat_0_oos,
        "p_hat_1": p_hat_1_oos,
        "mu_hat_0": mu_hat_0_oos,
        "mu_hat_1": mu_hat_1_oos,
    }


def compute_dlate_score_objects(data, nuisance_results, y_grid):
    """Compute point estimates and score objects for D-IV-LATE."""
    return compute_dlate_score_objects_common(
        data,
        nuisance_results,
        y_grid,
        epsilon=PROBABILITY_EPSILON,
    )


def dlate_estimator_empirical(data, nuisance_results, y_grid, *, return_diagnostics=False):
    """Estimate the D-LATE curve from the nuisance bundle."""
    score_objects = compute_dlate_score_objects(data, nuisance_results, y_grid)
    if return_diagnostics:
        return score_objects["dlate"], {
            "mean_psi_beta": score_objects["mean_psi_beta"],
            "abs_mean_psi_beta": score_objects["abs_mean_psi_beta"],
            "near_zero_denominator": score_objects["near_zero_denominator"],
        }
    return score_objects["dlate"]


def compute_first_stage_diagnostics(data, x_cols):
    """Compute simple LPM first-stage diagnostics for the instrument."""
    x_matrix = data[x_cols].to_numpy(dtype=float)
    z_values = data["Z"].to_numpy(dtype=float)
    w_values = data["W"].to_numpy(dtype=float)
    n_obs = len(data)

    intercept = np.ones((n_obs, 1))
    reduced = np.column_stack([intercept, x_matrix])
    full = np.column_stack([intercept, x_matrix, z_values])

    beta_reduced, *_ = np.linalg.lstsq(reduced, w_values, rcond=None)
    beta_full, *_ = np.linalg.lstsq(full, w_values, rcond=None)
    residual_reduced = w_values - reduced @ beta_reduced
    residual_full = w_values - full @ beta_full
    sse_reduced = float(np.dot(residual_reduced, residual_reduced))
    sse_full = float(np.dot(residual_full, residual_full))

    df_num = 1
    df_den = max(n_obs - full.shape[1], 1)
    if sse_full <= 0:
        first_stage_f = np.nan
    else:
        numerator = max(sse_reduced - sse_full, 0.0) / df_num
        denominator = sse_full / df_den
        first_stage_f = np.nan if denominator <= 0 else float(numerator / denominator)
    partial_r2 = np.nan if sse_reduced <= 0 else float(max(1.0 - sse_full / sse_reduced, 0.0))

    sample_first_stage = float(data.groupby("Z")["W"].mean().diff().iloc[-1])
    return {
        "sample_first_stage": sample_first_stage,
        "first_stage_f_stat": first_stage_f,
        "partial_r2": partial_r2,
    }


def compute_overlap_diagnostics(data, nuisance_results, x_cols, *, trim_bounds=(0.05, 0.95)):
    """Summarize overlap and balance diagnostics for the empirical pipeline."""
    lower, upper = trim_bounds
    pi_hat = clip_probabilities(nuisance_results["pi_hat"], epsilon=PROBABILITY_EPSILON)
    trim_mask = (pi_hat >= lower) & (pi_hat <= upper)

    balance_rows = []
    for column in x_cols:
        z0 = data.loc[data["Z"] == 0, column].to_numpy(dtype=float)
        z1 = data.loc[data["Z"] == 1, column].to_numpy(dtype=float)
        pooled_std = np.sqrt(0.5 * (np.var(z0, ddof=0) + np.var(z1, ddof=0)))
        if pooled_std <= 0:
            smd = 0.0
        else:
            smd = float((np.mean(z1) - np.mean(z0)) / pooled_std)
        balance_rows.append({"covariate": column, "standardized_mean_difference": smd})

    return {
        "pi_outside_05_95_share": float(1.0 - np.mean(trim_mask)),
        "pi_outside_05_95_count": int(np.sum(~trim_mask)),
        "pi_hat_min": float(np.min(pi_hat)),
        "pi_hat_max": float(np.max(pi_hat)),
        "pi_hat_p05": float(np.quantile(pi_hat, 0.05)),
        "pi_hat_p95": float(np.quantile(pi_hat, 0.95)),
        "balance_df": pd.DataFrame(balance_rows),
    }


def run_empirical_model(
    data,
    x_cols,
    y_grid,
    *,
    model_type,
    k_folds=EMPIRICAL_K_FOLDS,
    kan_steps=EMPIRICAL_KAN_STEPS,
    kan_config=None,
    rf_config=None,
):
    """Run one empirical nuisance-model specification and return curve plus diagnostics."""
    nuisance_results = estimate_nuisance_functions_empirical(
        data,
        x_cols,
        y_grid,
        model_type=model_type,
        k_folds=k_folds,
        kan_steps=kan_steps,
        kan_config=kan_config,
        rf_config=rf_config,
    )
    dlate_estimates, estimator_diagnostics = dlate_estimator_empirical(
        data,
        nuisance_results,
        y_grid,
        return_diagnostics=True,
    )
    first_stage_diagnostics = compute_first_stage_diagnostics(data, x_cols)
    overlap_diagnostics = compute_overlap_diagnostics(data, nuisance_results, x_cols)

    kan_config = dict(kan_config or {})
    kan_config.setdefault("steps", kan_steps)
    kan_config = build_kan_config(**kan_config)
    rf_config = build_rf_config(**(rf_config or {}))
    model_config_id = (
        build_kan_config_id(kan_config)
        if model_type == "kan"
        else build_rf_config_id(rf_config)
    )

    curve_df = pd.DataFrame(
        {
            "y_value": y_grid,
            "dlate_estimate": dlate_estimates,
            "model": model_type,
            "model_config_id": model_config_id,
        }
    )
    diagnostics = {
        "model": model_type,
        "model_config_id": model_config_id,
        **first_stage_diagnostics,
        "mean_psi_beta": estimator_diagnostics["mean_psi_beta"],
        "abs_mean_psi_beta": estimator_diagnostics["abs_mean_psi_beta"],
        "near_zero_denominator": int(estimator_diagnostics["near_zero_denominator"]),
        "pi_outside_05_95_share": overlap_diagnostics["pi_outside_05_95_share"],
        "pi_outside_05_95_count": overlap_diagnostics["pi_outside_05_95_count"],
        "pi_hat_min": overlap_diagnostics["pi_hat_min"],
        "pi_hat_max": overlap_diagnostics["pi_hat_max"],
        "pi_hat_p05": overlap_diagnostics["pi_hat_p05"],
        "pi_hat_p95": overlap_diagnostics["pi_hat_p95"],
    }
    return {
        "curve": curve_df,
        "diagnostics": diagnostics,
        "balance": overlap_diagnostics["balance_df"],
        "nuisance_results": nuisance_results,
    }


def save_empirical_curve_plot(curve_df, plot_path, *, title, ylabel):
    """Persist a single empirical curve plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(curve_df["y_value"], curve_df["dlate_estimate"], marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Net Financial Assets (y)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


def main(
    *,
    data_path=DEFAULT_DATA_PATH,
    results_dir=DEFAULT_RESULTS_DIR,
    y_grid_points=30,
    k_folds=EMPIRICAL_K_FOLDS,
    kan_steps=EMPIRICAL_KAN_STEPS,
):
    """
    Run the legacy single-model empirical KAN pipeline.

    This remains as a compatibility entrypoint for the existing smoke test and
    manuscript asset names. Comparative and robustness execution is handled by
    the wrapper script.
    """
    print("Starting D-LATE estimation for empirical application...")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data, x_cols = load_and_prepare_data(csv_path=data_path)
    if data.empty or len(data) < 100:
        print("Data loading or preprocessing resulted in insufficient data. Exiting.")
        return

    processed_data, _ = preprocess_empirical_data(data, x_cols, preprocess_mode="raw")
    y_grid = build_empirical_y_grid(processed_data, y_grid_points=y_grid_points)
    print(f"Data loaded: {len(processed_data)} observations after cleaning.")
    print(
        f"y_grid for D-LATE: from {float(y_grid.min()):.2f} to {float(y_grid.max()):.2f} "
        f"with {len(y_grid)} points."
    )

    print("Estimating nuisance functions (this may take a few minutes)...")
    outputs = run_empirical_model(
        processed_data,
        x_cols,
        y_grid,
        model_type="kan",
        k_folds=k_folds,
        kan_steps=kan_steps,
    )

    print("Estimating D-LATE...")
    results_df = outputs["curve"][["y_value", "dlate_estimate"]].copy()

    print("\n--- D-LATE Estimation Results ---")
    print(results_df)

    results_csv_path = results_dir / "empirical_kan-d-iv-late_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nEmpirical D-LATE results saved to {results_csv_path}")

    plot_path = results_dir / "empirical_kan-d-iv-late_plot.png"
    save_empirical_curve_plot(
        results_df,
        plot_path,
        title="Estimated Distributional LATE (KAN-D-LATE) - Pension Data",
        ylabel="KAN-D-LATE(y)",
    )
    print(f"Plot saved to {plot_path}")

    print("\n--- End of Empirical Application ---")


if __name__ == "__main__":
    main()
