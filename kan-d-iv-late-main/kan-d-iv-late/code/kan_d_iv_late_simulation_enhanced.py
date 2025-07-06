"""
Enhanced D-IV-LATE Simulation with Standardized KAN Implementation
This script implements the improvements suggested in the technical guide:
- Standardized KAN implementation using efficient_kan
- GPU acceleration for RTX-4000
- Bootstrap confidence intervals
- Multiple testing corrections
- Enhanced diagnostics and robustness checks
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import sys

# Import our standardized utilities
from kan_utils import (
    train_standardized_kan,
    bootstrap_dlate_ci,
    apply_multiple_testing_correction,
    dlate_point_se,  # Import the asymptotic SE function
    KAN_STEPS, K_FOLDS, DEVICE
)

# Enhanced simulation parameters
N_REPLICATIONS_FULL = 200
N_REPLICATIONS_FAST = 5 # Drastically reduced for hyperparameter search
N_SAMPLES = 1000
N_FEATURES = 5
Y_GRID_SIZE = 30
N_BOOTSTRAP = 50

# Hyperparameter grid for sensitivity analysis
HYPERPARAM_GRID = {
    'KAN_STEPS': [50, 150, 250],
    'KAN_HIDDEN_DIM': [16, 48, 64],
    'KAN_REG_STRENGTH': [1e-3, 1e-5]
}

print(f"Running enhanced simulation on {DEVICE}")
print(f"Configuration: {N_REPLICATIONS} replications, {N_SAMPLES} samples, K={K_FOLDS} folds")

def generate_dlate_data(n_samples=2000, n_features=5, seed=42):
    """
    Enhanced data generating process with complex non-linearities
    designed to test KAN vs RF performance
    """
    np.random.seed(seed)
    
    # Generate covariates with different distributions
    X = np.random.randn(n_samples, n_features)
    
    # Complex nuisance functions (favor KAN architecture)
    pi_x = torch.sigmoid(torch.tensor(
        X[:, 0] + np.sin(2*np.pi * X[:, 1]) - np.cos(np.pi * X[:, 2]) + 
        0.5 * X[:, 3]**2 - 0.3 * X[:, 4]**3
    )).numpy()
    Z = np.random.binomial(1, pi_x)
    
    # Treatment assignment with interaction effects
    p_zx = torch.sigmoid(torch.tensor(
        2*Z + X[:, 0] + np.sin(np.pi * X[:, 1]) - 0.5*np.cos(2*np.pi * X[:, 2]) + 
        0.3 * X[:, 3]**2 - 0.2 * X[:, 4]**2 + 0.1*Z*X[:, 0]
    )).numpy()
    W = np.random.binomial(1, p_zx)
    
    # Heterogeneous treatment effects
    treatment_effect = 8 + 2*np.cos(np.pi * X[:, 2]) + 0.5*X[:, 0] + np.random.randn(n_samples)*0.5
    
    # Potential outcomes with complex baseline
    Y0 = (X[:, 0] + 1.5*np.sin(np.pi * X[:, 1]) + 0.8*X[:, 2]**2 + 
          0.3*X[:, 3] - 0.2*X[:, 4]**2 + np.random.randn(n_samples))
    Y1 = Y0 + treatment_effect
    
    # Observed outcome
    Y = W * Y1 + (1 - W) * Y0
    
    # Create dataframe
    data = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
    data['Z'] = Z
    data['W'] = W
    data['Y'] = Y
    
    # True D-LATE function for evaluation
    def true_dlate_func(y):
        compliers = (Z == 1) & (W == 1) | (Z == 0) & (W == 0)  # Simplified complier definition
        if np.sum(compliers) == 0:
            return 0
        return np.mean(Y1[compliers] <= y) - np.mean(Y0[compliers] <= y)
    
    return data, true_dlate_func


def estimate_nuisance_functions_enhanced(data, y_grid, model_type='kan'):
    """
    Refactored nuisance function estimation for compatibility with asymptotic SEs.
    Estimates models for each instrument value Z=0 and Z=1 separately.
    """
    X_cols = [col for col in data.columns if col.startswith('X')]
    n_samples = len(data)
    
    # Initialize results dataframe
    results = pd.DataFrame(index=data.index)
    
    # Cross-fitting setup
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Initialize prediction columns
    results['pi_hat'] = np.nan
    results['p_hat_0'] = np.nan
    results['p_hat_1'] = np.nan
    for y_val in y_grid:
        results[f'mu_hat_0_{y_val}'] = np.nan
        results[f'mu_hat_1_{y_val}'] = np.nan

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
        # print(f"  Processing fold {fold_idx + 1}/{K_FOLDS}")
        
        data_train = data.iloc[train_idx]
        data_test = data.iloc[test_idx]
        X_test = data_test[X_cols].values

        # --- Model Type Dispatch ---
        if model_type == 'kan':
            def fit_predict(X_train, y_train, X_pred):
                model = train_standardized_kan(X_train, y_train)
                model.eval()
                with torch.no_grad():
                    X_pred_tensor = torch.FloatTensor(X_pred).to(DEVICE)
                    return torch.sigmoid(model(X_pred_tensor)).cpu().numpy().flatten()
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            def fit_predict(X_train, y_train, X_pred):
                # Handle cases with single class in training data
                if len(np.unique(y_train)) == 1:
                    return np.full(len(X_pred), y_train[0])
                model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
                model.fit(X_train, y_train)
                return model.predict_proba(X_pred)[:, 1]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # --- Nuisance Function Estimation ---
        
        # 1. pi_hat = E[Z | X]
        pi_hat_preds = fit_predict(data_train[X_cols].values, data_train['Z'].values, X_test)
        results.loc[test_idx, 'pi_hat'] = pi_hat_preds

        # 2. p_hat_z = E[W | Z=z, X] for z in {0, 1}
        for z_val in [0, 1]:
            train_z = data_train[data_train['Z'] == z_val]
            if len(train_z) > 10:
                p_hat_preds = fit_predict(train_z[X_cols].values, train_z['W'].values, X_test)
                results.loc[test_idx, f'p_hat_{z_val}'] = p_hat_preds
            else: # Fallback for sparse data
                results.loc[test_idx, f'p_hat_{z_val}'] = np.mean(data_train['W'])

        # 3. mu_hat_z = E[1(Y<=y) | Z=z, X] for z in {0, 1} and y in y_grid
        for y_val in y_grid:
            y_binary_train = (data_train['Y'] <= y_val).astype(int)
            for z_val in [0, 1]:
                train_z = data_train[data_train['Z'] == z_val]
                y_binary_train_z = (train_z['Y'] <= y_val).astype(int)
                if len(train_z) > 10:
                    mu_hat_preds = fit_predict(train_z[X_cols].values, y_binary_train_z.values, X_test)
                    results.loc[test_idx, f'mu_hat_{z_val}_{y_val}'] = mu_hat_preds
                else: # Fallback for sparse data
                    results.loc[test_idx, f'mu_hat_{z_val}_{y_val}'] = np.mean(y_binary_train)
    
    return results


def dlate_estimator(data, nuisance_df, y_grid):
    """
    Computes the D-LATE using the Double/Debiased Machine Learning (DML)
    influence function for the Local Average Treatment Effect (LATE).
    This version is simplified to return only point estimates for bootstrap compatibility.
    """
    Z, W, Y = data['Z'].values, data['W'].values, data['Y'].values
    
    # Ensure that nuisance_df has the required columns from the enhanced function
    # The enhanced function provides p_hat_0 and p_hat_1 directly
    pi_hat = nuisance_df['pi_hat'].values
    p_hat_0 = nuisance_df['p_hat_0'].values
    p_hat_1 = nuisance_df['p_hat_1'].values

    psi_beta = (p_hat_1 - p_hat_0) + (Z / pi_hat) * (W - p_hat_1) - ((1 - Z) / (1 - pi_hat)) * (W - p_hat_0)
    E_psi_beta = np.mean(psi_beta)

    if np.abs(E_psi_beta) < 1e-8:
        return np.full(len(y_grid), np.nan) # Return NaNs if denominator is zero

    dlate_estimates = []
    for y_val in y_grid:
        mu_hat_0_y = nuisance_df[f'mu_hat_0_{y_val}'].values
        mu_hat_1_y = nuisance_df[f'mu_hat_1_{y_val}'].values
        y_indicator = (Y <= y_val).astype(int)

        psi_alpha = (mu_hat_1_y - mu_hat_0_y) + \
                    (Z / pi_hat) * (y_indicator - mu_hat_1_y) - \
                    ((1 - Z) / (1 - pi_hat)) * (y_indicator - mu_hat_0_y)
        
        E_psi_alpha = np.mean(psi_alpha)
        dlate_estimates.append(E_psi_alpha / E_psi_beta)

    return np.array(dlate_estimates)


def run_enhanced_simulation(use_bootstrap=True, n_replications=N_REPLICATIONS_FULL):
    """
    Run the enhanced Monte Carlo simulation.
    Can be run in 'full' mode with bootstrapping or 'fast' mode without.
    """
    print(f"Starting Simulation (Bootstrap: {use_bootstrap}, Replications: {n_replications})")
    print("=" * 50)
    
    y_grid = np.linspace(-8, 15, Y_GRID_SIZE)
    results_storage = {'kan': [], 'rf': []}
    true_values = []
    
    for rep in tqdm(range(n_replications), desc="Monte Carlo Replications"):
        data, true_dlate_func = generate_dlate_data(n_samples=N_SAMPLES, n_features=N_FEATURES, seed=rep)
        true_dlate = [true_dlate_func(y) for y in y_grid]
        true_values.append(true_dlate)
        
        for model_type in ['kan', 'rf']:
            try:
                nuisance_df = estimate_nuisance_functions_enhanced(data, y_grid, model_type=model_type)
                
                if use_bootstrap:
                    dlate_results = bootstrap_dlate_ci(
                        data, y_grid,
                        nuisance_estimator=lambda d, y: estimate_nuisance_functions_enhanced(d, y, model_type=model_type),
                        dlate_estimator=dlate_estimator,
                        n_bootstrap=N_BOOTSTRAP
                    )
                else: # Fast mode for hyperparameter tuning
                    dlate_est = dlate_estimator(data, nuisance_df, y_grid)
                    dlate_results = {
                        'point_estimates': dlate_est,
                        'ci_lower': np.full(len(y_grid), np.nan),
                        'ci_upper': np.full(len(y_grid), np.nan)
                    }
                results_storage[model_type].append(dlate_results)

            except Exception as e:
                print(f"{model_type.upper()} estimation failed in replication {rep}: {e}")
                nan_results = {
                    'point_estimates': np.full(len(y_grid), np.nan),
                    'ci_lower': np.full(len(y_grid), np.nan),
                    'ci_upper': np.full(len(y_grid), np.nan)
                }
                results_storage[model_type].append(nan_results)

    # --- Process and Aggregate Results ---
    true_values = np.array(true_values)
    
    # Create results dataframe
    results_df = pd.DataFrame({'y_value': y_grid, 'true_dlate': np.mean(true_values, axis=0)})
    
    # Extract results from bootstrap output
    kan_estimates = np.array([r['point_estimates'] for r in results_storage['kan']])
    rf_estimates = np.array([r['point_estimates'] for r in results_storage['rf']])

    for model_type, estimates in [('kan', kan_estimates), ('rf', rf_estimates)]:
        # Performance metrics
        results_df[f'{model_type}_estimate'] = np.nanmean(estimates, axis=0)
        results_df[f'{model_type}_bias'] = np.nanmean(estimates - true_values, axis=0)
        results_df[f'{model_type}_rmse'] = np.sqrt(np.nanmean((estimates - true_values)**2, axis=0))
        results_df[f'{model_type}_std'] = np.nanstd(estimates, axis=0)
        
        # Confidence interval coverage
        ci_lower = np.array([r['ci_lower'] for r in results_storage[model_type]])
        ci_upper = np.array([r['ci_upper'] for r in results_storage[model_type]])
        
        results_df[f'{model_type}_ci_lower_mean'] = np.nanmean(ci_lower, axis=0)
        results_df[f'{model_type}_ci_upper_mean'] = np.nanmean(ci_upper, axis=0)
        
        coverage = np.mean((true_values >= ci_lower) & (true_values <= ci_upper), axis=0)
        results_df[f'{model_type}_coverage'] = coverage

    # Statistical tests for differences in RMSE
    p_values = []
    for i in range(len(y_grid)):
        from scipy.stats import ttest_rel
        kan_abs_error = np.abs(kan_estimates[:, i] - true_values[:, i])
        rf_abs_error = np.abs(rf_estimates[:, i] - true_values[:, i])
        valid_mask = ~(np.isnan(kan_abs_error) | np.isnan(rf_abs_error))
        
        if np.sum(valid_mask) > 10:
            # Test if KAN error is significantly smaller than RF error
            # Note: This test might be less meaningful now with bootstrap CIs, but we keep it for comparison
            _, p_val = ttest_rel(kan_abs_error[valid_mask], rf_abs_error[valid_mask], alternative='less')
            p_values.append(p_val)
        else:
            p_values.append(1.0)
    
    # Multiple testing correction
    mt_results = apply_multiple_testing_correction(p_values, method='fdr_bh')
    results_df['p_value_rmse_diff'] = p_values
    results_df['p_corrected_rmse_diff'] = mt_results['p_corrected']
    results_df['kan_significantly_better'] = mt_results['rejected']
    
    return results_df


def create_enhanced_plots(results_df, output_dir='../results'):
    """
    Create comprehensive plots of simulation results, now with confidence bands.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Enhanced Simulation Results: KAN vs. Random Forest', fontsize=16)
    
    # Plot 1: Estimates vs True Values with Confidence Bands
    ax = axes[0, 0]
    ax.plot(results_df['y_value'], results_df['true_dlate'], 'k-', linewidth=2.5, label='True D-LATE', zorder=5)
    ax.plot(results_df['y_value'], results_df['kan_estimate'], 'r--', linewidth=2, label='KAN Estimate')
    ax.fill_between(
        results_df['y_value'], 
        results_df['kan_ci_lower_mean'], 
        results_df['kan_ci_upper_mean'], 
        color='red', alpha=0.2, label='KAN 95% CI'
    )
    ax.plot(results_df['y_value'], results_df['rf_estimate'], 'b:', linewidth=2, label='RF Estimate')
    ax.fill_between(
        results_df['y_value'], 
        results_df['rf_ci_lower_mean'], 
        results_df['rf_ci_upper_mean'], 
        color='blue', alpha=0.15, label='RF 95% CI'
    )
    ax.set_xlabel('y')
    ax.set_ylabel('D-LATE')
    ax.set_title('D-LATE Estimates with 95% Confidence Bands')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Bias Comparison
    ax = axes[0, 1]
    ax.plot(results_df['y_value'], results_df['kan_bias'], 'r-', linewidth=2, label='KAN Bias')
    ax.plot(results_df['y_value'], results_df['rf_bias'], 'b-', linewidth=2, label='RF Bias')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('y')
    ax.set_ylabel('Bias')
    ax.set_title('Bias Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RMSE Comparison
    ax = axes[1, 0]
    ax.plot(results_df['y_value'], results_df['kan_rmse'], 'r-', linewidth=2, label='KAN RMSE')
    ax.plot(results_df['y_value'], results_df['rf_rmse'], 'b-', linewidth=2, label='RF RMSE')
    ax.set_xlabel('y')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error (RMSE) Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistical Significance of RMSE Difference
    ax = axes[1, 1]
    significant_y = results_df['y_value'][results_df['kan_significantly_better']]
    ax.plot(results_df['y_value'], -np.log10(results_df['p_corrected_rmse_diff']), 'g-', linewidth=2)
    ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.7, label='α = 0.05')
    ax.scatter(significant_y, -np.log10(results_df.loc[results_df['kan_significantly_better'], 'p_corrected_rmse_diff']), 
               color='red', s=50, alpha=0.7, zorder=5, label='KAN Significantly Better')
    ax.set_xlabel('y')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Significance of RMSE Difference (FDR Corrected)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'enhanced_simulation_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Enhanced simulation plots saved to {output_dir}")


if __name__ == "__main__":
    # --- Fast Hyperparameter Sensitivity Analysis ---
    print("Starting Hyperparameter Sensitivity Analysis (Fast Mode)")
    all_hyperparam_results = []

    for steps in HYPERPARAM_GRID['KAN_STEPS']:
        for hidden_dim in HYPERPARAM_GRID['KAN_HIDDEN_DIM']:
            for reg_strength in HYPERPARAM_GRID['KAN_REG_STRENGTH']:
                print(f"\n--- Testing KAN_STEPS={steps}, KAN_HIDDEN_DIM={hidden_dim}, KAN_REG_STRENGTH={reg_strength} ---")
                
                # Temporarily override KAN settings in kan_utils
                from kan_utils import KAN_STEPS, KAN_HIDDEN_DIM, KAN_REG_STRENGTH
                original_kan_steps, original_kan_hidden_dim, original_kan_reg_strength = KAN_STEPS, KAN_HIDDEN_DIM, KAN_REG_STRENGTH
                import kan_utils
                kan_utils.KAN_STEPS, kan_utils.KAN_HIDDEN_DIM, kan_utils.KAN_REG_STRENGTH = steps, hidden_dim, reg_strength

                # Run simulation in fast mode (no bootstrap)
                results_df = run_enhanced_simulation(use_bootstrap=False, n_replications=N_REPLICATIONS_FAST)
                
                # Store primary metric (RMSE)
                mean_rmse = np.mean(results_df['kan_rmse'])
                all_hyperparam_results.append({
                    'steps': steps, 'hidden_dim': hidden_dim, 'reg_strength': reg_strength, 'mean_rmse': mean_rmse
                })

                # Restore original settings
                kan_utils.KAN_STEPS, kan_utils.KAN_HIDDEN_DIM, kan_utils.KAN_REG_STRENGTH = original_kan_steps, original_kan_hidden_dim, original_kan_reg_strength

    # --- Process and Save Hyperparameter Results ---
    hyperparam_results_df = pd.DataFrame(all_hyperparam_results)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    hyperparam_results_path = os.path.join(output_dir, 'hyperparameter_tuning_results.csv')
    hyperparam_results_df.to_csv(hyperparam_results_path, index=False)
    
    print("\n" + "="*50)
    print("Hyperparameter Sensitivity Analysis Complete")
    print(f"Results saved to {hyperparam_results_path}")
    print("="*50)
    print(hyperparam_results_df.sort_values('mean_rmse'))
    print("="*50)
    
    # Find and report the best parameters
    best_params = hyperparam_results_df.loc[hyperparam_results_df['mean_rmse'].idxmin()]
    print("\nBest performing hyperparameters based on RMSE:")
    print(best_params)
    print("\nTo run a full simulation with these parameters, update kan_utils.py and run this script again.")
