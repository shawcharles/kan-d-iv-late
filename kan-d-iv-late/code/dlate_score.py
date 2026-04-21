from __future__ import annotations

import numpy as np


def _get_required_array(nuisance_results, *keys):
    for key in keys:
        if key in nuisance_results:
            return np.asarray(nuisance_results[key], dtype=float)
    raise KeyError(f"Missing nuisance result. Expected one of: {keys}")


def compute_dlate_score_objects(data, nuisance_results, y_grid, *, epsilon=1e-6):
    """Compute D-IV-LATE point estimates and score objects from level scores."""
    z_values = data["Z"].to_numpy(dtype=float)
    w_values = data["W"].to_numpy(dtype=float)
    y_values = data["Y"].to_numpy(dtype=float)

    pi_hat = np.clip(
        _get_required_array(nuisance_results, "pi_hat"),
        epsilon,
        1 - epsilon,
    )
    p_hat_0 = np.clip(
        _get_required_array(nuisance_results, "p_hat_0", "p_hat_z0"),
        epsilon,
        1 - epsilon,
    )
    p_hat_1 = np.clip(
        _get_required_array(nuisance_results, "p_hat_1", "p_hat_z1"),
        epsilon,
        1 - epsilon,
    )
    mu_hat_0 = np.clip(
        _get_required_array(nuisance_results, "mu_hat_0", "mu_hat_z0"),
        epsilon,
        1 - epsilon,
    )
    mu_hat_1 = np.clip(
        _get_required_array(nuisance_results, "mu_hat_1", "mu_hat_z1"),
        epsilon,
        1 - epsilon,
    )

    psi_beta = (
        (p_hat_1 - p_hat_0)
        + (z_values / pi_hat) * (w_values - p_hat_1)
        - ((1.0 - z_values) / (1.0 - pi_hat)) * (w_values - p_hat_0)
    )
    mean_psi_beta = float(np.mean(psi_beta))
    abs_mean_psi_beta = float(np.abs(mean_psi_beta))

    dlate = []
    psi_alpha_columns = []
    for y_idx, y_val in enumerate(y_grid):
        y_indicator = (y_values <= y_val).astype(float)
        psi_alpha = (
            (mu_hat_1[:, y_idx] - mu_hat_0[:, y_idx])
            + (z_values / pi_hat) * (y_indicator - mu_hat_1[:, y_idx])
            - ((1.0 - z_values) / (1.0 - pi_hat)) * (y_indicator - mu_hat_0[:, y_idx])
        )
        psi_alpha_columns.append(psi_alpha)
        if abs_mean_psi_beta < epsilon:
            dlate.append(np.nan)
        else:
            dlate.append(float(np.mean(psi_alpha) / mean_psi_beta))

    return {
        "dlate": np.asarray(dlate, dtype=float),
        "psi_alpha": np.column_stack(psi_alpha_columns),
        "psi_beta": np.asarray(psi_beta, dtype=float),
        "mean_psi_beta": mean_psi_beta,
        "abs_mean_psi_beta": abs_mean_psi_beta,
        "near_zero_denominator": bool(abs_mean_psi_beta < epsilon),
    }
