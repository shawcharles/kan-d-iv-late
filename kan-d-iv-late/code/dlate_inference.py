from __future__ import annotations

import numpy as np

from dlate_score import compute_dlate_score_objects


def dlate_asymptotic_inference(data, nuisance_results, y_grid, *, alpha=0.05, epsilon=1e-6):
    """Compute pointwise asymptotic standard errors and Wald intervals."""
    score_objects = compute_dlate_score_objects(data, nuisance_results, y_grid, epsilon=epsilon)
    dlate = score_objects["dlate"]
    psi_alpha = score_objects["psi_alpha"]
    psi_beta = score_objects["psi_beta"]
    mean_psi_beta = score_objects["mean_psi_beta"]
    n_obs = len(data)

    z_value = 1.959963984540054
    if alpha != 0.05:
        # Normal-approximation constant for common two-sided levels.
        from statistics import NormalDist

        z_value = NormalDist().inv_cdf(1 - alpha / 2)

    standard_errors = np.full(len(y_grid), np.nan, dtype=float)
    ci_lower = np.full(len(y_grid), np.nan, dtype=float)
    ci_upper = np.full(len(y_grid), np.nan, dtype=float)

    if np.abs(mean_psi_beta) < epsilon:
        return {
            "point_estimates": dlate,
            "standard_errors": standard_errors,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean_psi_beta": mean_psi_beta,
            "near_zero_denominator": True,
        }

    for y_idx in range(len(y_grid)):
        influence_values = (psi_alpha[:, y_idx] - dlate[y_idx] * psi_beta) / mean_psi_beta
        variance = float(np.var(influence_values, ddof=1) / n_obs)
        se = float(np.sqrt(max(variance, 0.0)))
        standard_errors[y_idx] = se
        ci_lower[y_idx] = dlate[y_idx] - z_value * se
        ci_upper[y_idx] = dlate[y_idx] + z_value * se

    return {
        "point_estimates": dlate,
        "standard_errors": standard_errors,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_psi_beta": mean_psi_beta,
        "near_zero_denominator": False,
    }


def bootstrap_dlate_inference(
    data,
    y_grid,
    nuisance_estimator,
    *,
    alpha=0.05,
    n_bootstrap=50,
    random_state=0,
    epsilon=1e-6,
):
    """Compute percentile bootstrap D-IV-LATE intervals."""
    rng = np.random.default_rng(random_state)
    bootstrap_estimates = []
    n_obs = len(data)

    for _ in range(n_bootstrap):
        bootstrap_indices = rng.integers(0, n_obs, size=n_obs)
        bootstrap_data = data.iloc[bootstrap_indices].reset_index(drop=True)
        nuisance_results = nuisance_estimator(bootstrap_data, y_grid)
        score_objects = compute_dlate_score_objects(
            bootstrap_data,
            nuisance_results,
            y_grid,
            epsilon=epsilon,
        )
        bootstrap_estimates.append(score_objects["dlate"])

    bootstrap_estimates = np.asarray(bootstrap_estimates, dtype=float)
    lower = np.nanpercentile(bootstrap_estimates, 100 * (alpha / 2), axis=0)
    upper = np.nanpercentile(bootstrap_estimates, 100 * (1 - alpha / 2), axis=0)
    point_estimates = np.nanmean(bootstrap_estimates, axis=0)
    return {
        "point_estimates": point_estimates,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_estimates": bootstrap_estimates,
    }


def summarize_interval_coverage(point_estimates, ci_lower, ci_upper, true_values):
    """Return pointwise coverage flags and interval widths."""
    covers = (ci_lower <= true_values) & (true_values <= ci_upper)
    widths = ci_upper - ci_lower
    return {
        "covers": np.asarray(covers, dtype=bool),
        "widths": np.asarray(widths, dtype=float),
        "point_estimates": np.asarray(point_estimates, dtype=float),
    }
