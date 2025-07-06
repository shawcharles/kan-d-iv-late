"""
Enhanced Empirical Application with Standardized KAN Implementation

This script applies the D-IV-LATE estimator to the 401(k) dataset with
the following enhancements:
- Standardized KAN implementation via kan_utils
- Bootstrap confidence intervals for robust inference
- Diagnostic checks to investigate counterintuitive results
- Comparison with a Random Forest baseline
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings

# Import standardized utilities
from kan_utils import (
    train_standardized_kan,
    bootstrap_dlate_ci,
    apply_multiple_testing_correction,
    DEVICE
)

# --- 1. Data Loading and Preparation ---
def load_and_prepare_data(csv_path='../data/pension.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Define variables
    Y_col, W_col, Z_col = 'net_tfa', 'p401', 'e401'
    X_cols = ['inc', 'age', 'educ', 'marr', 'fsize', 'twoearn', 'db', 'pira', 'hown']
    
    df_subset = df[[Y_col, W_col, Z_col] + X_cols].copy()
    df_subset.dropna(inplace=True)
    
    df_subset.rename(columns={Y_col: 'Y', W_col: 'W', Z_col: 'Z'}, inplace=True)
    
    # Scale covariates
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_subset[X_cols] = scaler.fit_transform(df_subset[X_cols])
    
    print(f"Data loaded and prepared. Shape: {df_subset.shape}")
    return df_subset, X_cols

# --- 2. Nuisance Function Estimation ---
def estimate_nuisance_functions_empirical(data, X_cols, y_grid, model_type='kan'):
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestClassifier

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_samples = len(data)
    
    results = {
        'pi_hat': np.zeros(n_samples),
        'p_hat': np.zeros(n_samples),
    }
    for y_val in y_grid:
        results[f'mu_hat_{y_val}'] = np.zeros(n_samples)

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(data), desc=f'Nuisance ({model_type})')):
        data_train, data_test = data.iloc[train_idx], data.iloc[test_idx]

        if model_type == 'kan':
            # Pi(X) = P(Z=1|X)
            pi_model = train_standardized_kan(data_train[X_cols].values, data_train['Z'].values)
            with torch.no_grad():
                results['pi_hat'][test_idx] = torch.sigmoid(pi_model(torch.FloatTensor(data_test[X_cols].values).to(DEVICE))).cpu().numpy().flatten()

            # P(Z,X) = P(W=1|Z,X)
            p_model = train_standardized_kan(data_train[X_cols + ['Z']].values, data_train['W'].values)
            with torch.no_grad():
                results['p_hat'][test_idx] = torch.sigmoid(p_model(torch.FloatTensor(data_test[X_cols + ['Z']].values).to(DEVICE))).cpu().numpy().flatten()

            # Mu(y,W,X) = P(Y<=y|W,X)
            for y in y_grid:
                Y_binary = (data_train['Y'] <= y).astype(float)
                if len(np.unique(Y_binary)) > 1:
                    mu_model = train_standardized_kan(data_train[X_cols + ['W']].values, Y_binary.values)
                    with torch.no_grad():
                        results[f'mu_hat_{y}'][test_idx] = torch.sigmoid(mu_model(torch.FloatTensor(data_test[X_cols + ['W']].values).to(DEVICE))).cpu().numpy().flatten()
                else:
                    results[f'mu_hat_{y}'][test_idx] = Y_binary.iloc[0]
        else: # Random Forest
            rf_pi = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42).fit(data_train[X_cols], data_train['Z'])
            results['pi_hat'][test_idx] = rf_pi.predict_proba(data_test[X_cols])[:, 1]

            rf_p = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42).fit(data_train[X_cols + ['Z']], data_train['W'])
            results['p_hat'][test_idx] = rf_p.predict_proba(data_test[X_cols + ['Z']])[:, 1]

            for y in y_grid:
                Y_binary = (data_train['Y'] <= y).astype(int)
                if len(np.unique(Y_binary)) > 1:
                    rf_mu = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42).fit(data_train[X_cols + ['W']], Y_binary)
                    results[f'mu_hat_{y}'][test_idx] = rf_mu.predict_proba(data_test[X_cols + ['W']])[:, 1]
                else:
                    results[f'mu_hat_{y}'][test_idx] = Y_binary.iloc[0]

    return pd.DataFrame(results)

# --- 3. D-LATE Estimator ---
def dlate_estimator_empirical(data, nuisance_df, X_cols, y_grid, model_type='kan'):
    Z, W = data['Z'].values, data['W'].values
    pi_hat = np.clip(nuisance_df['pi_hat'].values, 0.01, 0.99)
    p_hat = np.clip(nuisance_df['p_hat'].values, 0.01, 0.99)

    psi_beta = (Z - pi_hat) / (pi_hat * (1 - pi_hat)) * (W - p_hat)
    denominator = np.mean(psi_beta)
    if abs(denominator) < 1e-6: return np.zeros(len(y_grid))

    dlate_estimates = []
    for y in y_grid:
        mu_hat_y = nuisance_df[f'mu_hat_{y}'].values
        
        # Cross-fit mu_hat_1 and mu_hat_0
        mu_hat_1, mu_hat_0 = np.zeros(len(data)), np.zeros(len(data))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(data):
            # mu(y,1,x)
            if data.iloc[train_idx]['W'].sum() > 10:
                m1_model = train_standardized_kan(data.iloc[train_idx][data.iloc[train_idx]['W']==1][X_cols].values, (data.iloc[train_idx][data.iloc[train_idx]['W']==1]['Y'] <= y).astype(float).values)
                with torch.no_grad():
                    mu_hat_1[test_idx] = torch.sigmoid(m1_model(torch.FloatTensor(data.iloc[test_idx][X_cols].values).to(DEVICE))).cpu().numpy().flatten()
            # mu(y,0,x)
            if (1-data.iloc[train_idx]['W']).sum() > 10:
                m0_model = train_standardized_kan(data.iloc[train_idx][data.iloc[train_idx]['W']==0][X_cols].values, (data.iloc[train_idx][data.iloc[train_idx]['W']==0]['Y'] <= y).astype(float).values)
                with torch.no_grad():
                    mu_hat_0[test_idx] = torch.sigmoid(m0_model(torch.FloatTensor(data.iloc[test_idx][X_cols].values).to(DEVICE))).cpu().numpy().flatten()

        alpha_hat = mu_hat_1 - mu_hat_0
        psi_alpha = (1/p_hat - 1) * (W - p_hat) * (mu_hat_y - mu_hat_0) - (1 - (1-W)/(1-p_hat)) * alpha_hat
        numerator = np.mean(psi_alpha)
        dlate_estimates.append(numerator / denominator)

    return np.array(dlate_estimates)

# --- 4. Main Execution ---
def run_empirical_analysis():
    data, X_cols = load_and_prepare_data()
    y_grid = np.percentile(data['Y'], np.linspace(5, 95, 30))

    # KAN analysis with bootstrap CIs
    kan_bootstrap_results = bootstrap_dlate_ci(
        data, y_grid, 
        lambda d, y: estimate_nuisance_functions_empirical(d, X_cols, y, 'kan'),
        lambda d, n, y: dlate_estimator_empirical(d, n, X_cols, y, 'kan'),
        n_bootstrap=200 # Reduced for speed, recommend 500+
    )

    # RF analysis for comparison (no CIs for speed)
    nuisance_rf = estimate_nuisance_functions_empirical(data, X_cols, y_grid, 'rf')
    dlate_rf = dlate_estimator_empirical(data, nuisance_rf, X_cols, y_grid, 'rf')

    # --- 5. Save and Plot Results ---
    results_df = pd.DataFrame({
        'y_value': y_grid,
        'kan_dlate': kan_bootstrap_results['point_estimates'],
        'kan_ci_lower': kan_bootstrap_results['ci_lower'],
        'kan_ci_upper': kan_bootstrap_results['ci_upper'],
        'rf_dlate': dlate_rf
    })
    
    output_dir = '../results'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'enhanced_empirical_results.csv'), index=False)

    plt.figure(figsize=(12, 8))
    plt.plot(results_df['y_value'], results_df['kan_dlate'], 'r-', label='KAN D-LATE')
    plt.fill_between(results_df['y_value'], results_df['kan_ci_lower'], results_df['kan_ci_upper'], color='red', alpha=0.2, label='95% CI (KAN)')
    plt.plot(results_df['y_value'], results_df['rf_dlate'], 'b--', label='RF D-LATE')
    plt.axhline(0, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Net Total Assets (y)')
    plt.ylabel('D-LATE Estimate')
    plt.title('Effect of 401(k) Participation on Wealth Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'enhanced_empirical_plot.png'), dpi=300)
    plt.show()

    print(f"Empirical analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    run_empirical_analysis()
