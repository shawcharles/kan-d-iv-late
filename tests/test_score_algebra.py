import numpy as np
import pandas as pd

from kan_test_helpers import ensure_stub_efficient_kan

ensure_stub_efficient_kan()

from kan_d_iv_late import dlate_score as score_module
from kan_d_iv_late import empirical


def test_dlate_estimator_matches_manual_score_formula():
    data = pd.DataFrame(
        {
            "Z": [0, 1],
            "W": [0, 1],
            "Y": [0.0, 1.0],
        }
    )
    nuisance_results = {
        "pi_hat": np.array([0.5, 0.5]),
        "p_hat_0": np.array([0.2, 0.2]),
        "p_hat_1": np.array([0.8, 0.8]),
        "mu_hat_0": np.array([[0.2], [0.2]]),
        "mu_hat_1": np.array([[0.6], [0.6]]),
    }

    result = empirical.dlate_estimator_empirical(data, nuisance_results, np.array([0.5]))
    assert result.shape == (1,)
    assert np.isclose(result[0], -1.0)


def test_dlate_estimator_returns_nan_when_denominator_collapses():
    data = pd.DataFrame(
        {
            "Z": [0, 1],
            "W": [0, 0],
            "Y": [0.0, 1.0],
        }
    )
    nuisance_results = {
        "pi_hat": np.array([0.5, 0.5]),
        "p_hat_0": np.array([0.0, 0.0]),
        "p_hat_1": np.array([0.0, 0.0]),
        "mu_hat_0": np.array([[0.2], [0.2]]),
        "mu_hat_1": np.array([[0.6], [0.6]]),
    }

    result = empirical.dlate_estimator_empirical(data, nuisance_results, np.array([0.5]))
    assert np.isnan(result[0])


def test_level_scores_recover_population_targets_under_exact_nuisances():
    rng = np.random.default_rng(123)
    n_obs = 200000

    x_values = rng.binomial(1, 0.5, size=n_obs)
    pi = np.where(x_values == 0, 0.4, 0.6)
    z_values = rng.binomial(1, pi)

    p_hat_0 = np.where(x_values == 0, 0.10, 0.20)
    p_hat_1 = np.where(x_values == 0, 0.40, 0.50)
    w_values = rng.binomial(1, np.where(z_values == 1, p_hat_1, p_hat_0))

    prob_y_one_z0 = np.where(x_values == 0, 0.30, 0.40)
    prob_y_one_z1 = np.where(x_values == 0, 0.70, 0.80)
    y_values = rng.binomial(1, np.where(z_values == 1, prob_y_one_z1, prob_y_one_z0))

    data = pd.DataFrame({"Z": z_values, "W": w_values, "Y": y_values})
    nuisance_results = {
        "pi_hat": pi,
        "p_hat_0": p_hat_0,
        "p_hat_1": p_hat_1,
        "mu_hat_0": (1.0 - prob_y_one_z0).reshape(-1, 1),
        "mu_hat_1": (1.0 - prob_y_one_z1).reshape(-1, 1),
    }
    score_objects = score_module.compute_dlate_score_objects(
        data,
        nuisance_results,
        np.array([0.5]),
    )

    true_beta = float(np.mean(p_hat_1 - p_hat_0))
    true_alpha = float(np.mean((1.0 - prob_y_one_z1) - (1.0 - prob_y_one_z0)))

    assert np.isclose(score_objects["mean_psi_beta"], true_beta, atol=0.01)
    assert np.isclose(np.mean(score_objects["psi_alpha"][:, 0]), true_alpha, atol=0.01)
    assert np.isclose(score_objects["dlate"][0], true_alpha / true_beta, atol=0.03)
