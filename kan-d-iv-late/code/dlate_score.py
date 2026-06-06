from __future__ import annotations

import numpy as np


REQUIRED_DATA_COLUMNS = ("Z", "W", "Y")


def _get_required_array(nuisance_results, *keys):
    for key in keys:
        if key in nuisance_results:
            return np.asarray(nuisance_results[key], dtype=float)
    raise KeyError(f"Missing nuisance result. Expected one of: {keys}")


def _validate_epsilon(epsilon):
    epsilon_array = np.asarray(epsilon, dtype=float)
    if epsilon_array.shape != ():
        raise ValueError("epsilon must be a positive finite scalar")

    epsilon_value = float(epsilon_array)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0 or epsilon_value >= 0.5:
        raise ValueError("epsilon must be a positive finite scalar less than 0.5")
    return epsilon_value


def _validate_y_grid(y_grid):
    y_grid_array = np.asarray(y_grid, dtype=float)
    if y_grid_array.ndim != 1:
        raise ValueError(f"y_grid must be one-dimensional; got shape {y_grid_array.shape}")
    if y_grid_array.size == 0:
        raise ValueError("y_grid must contain at least one value")
    if not np.all(np.isfinite(y_grid_array)):
        raise ValueError("y_grid must contain only finite values")
    return y_grid_array


def _validate_data_columns(data):
    missing_columns = [column for column in REQUIRED_DATA_COLUMNS if column not in data]
    if missing_columns:
        raise KeyError(f"Missing required data columns: {missing_columns}")


def _get_data_column(data, column, n_obs):
    if hasattr(data[column], "to_numpy"):
        values = data[column].to_numpy(dtype=float)
    else:
        values = np.asarray(data[column], dtype=float)

    if values.shape != (n_obs,):
        raise ValueError(f"Data column {column!r} must have shape ({n_obs},); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Data column {column!r} must contain only finite values")
    return values


def _validate_vector_nuisance(values, name, n_obs):
    if values.shape != (n_obs,):
        raise ValueError(f"Nuisance {name!r} must have shape ({n_obs},); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nuisance {name!r} must contain only finite values")
    return values


def _validate_matrix_nuisance(values, name, n_obs, n_y):
    expected_shape = (n_obs, n_y)
    if values.shape != expected_shape:
        raise ValueError(f"Nuisance {name!r} must have shape {expected_shape}; got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nuisance {name!r} must contain only finite values")
    return values


def compute_dlate_score_objects(data, nuisance_results, y_grid, *, epsilon=1e-6):
    """Compute D-IV-LATE point estimates and score objects from level scores."""
    epsilon = _validate_epsilon(epsilon)
    y_grid = _validate_y_grid(y_grid)
    n_obs = len(data)
    n_y = len(y_grid)
    _validate_data_columns(data)

    z_values = _get_data_column(data, "Z", n_obs)
    w_values = _get_data_column(data, "W", n_obs)
    y_values = _get_data_column(data, "Y", n_obs)

    pi_hat = np.clip(
        _validate_vector_nuisance(
            _get_required_array(nuisance_results, "pi_hat"),
            "pi_hat",
            n_obs,
        ),
        epsilon,
        1 - epsilon,
    )
    p_hat_0 = np.clip(
        _validate_vector_nuisance(
            _get_required_array(nuisance_results, "p_hat_0", "p_hat_z0"),
            "p_hat_0",
            n_obs,
        ),
        epsilon,
        1 - epsilon,
    )
    p_hat_1 = np.clip(
        _validate_vector_nuisance(
            _get_required_array(nuisance_results, "p_hat_1", "p_hat_z1"),
            "p_hat_1",
            n_obs,
        ),
        epsilon,
        1 - epsilon,
    )
    mu_hat_0 = np.clip(
        _validate_matrix_nuisance(
            _get_required_array(nuisance_results, "mu_hat_0", "mu_hat_z0"),
            "mu_hat_0",
            n_obs,
            n_y,
        ),
        epsilon,
        1 - epsilon,
    )
    mu_hat_1 = np.clip(
        _validate_matrix_nuisance(
            _get_required_array(nuisance_results, "mu_hat_1", "mu_hat_z1"),
            "mu_hat_1",
            n_obs,
            n_y,
        ),
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
