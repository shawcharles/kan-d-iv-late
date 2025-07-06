import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import torch
from sklearn.preprocessing import StandardScaler
from kan_utils import train_standardized_kan, DEVICE, MIN_CLASS_COUNT_THRESHOLD

# --- 1. Load and Prepare Data ---
def load_and_prepare_data(csv_path='../data/pension.csv'):
    """
    Loads the pension dataset, selects relevant columns, and performs basic preprocessing.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return pd.DataFrame(), []

    df = pd.read_csv(csv_path)
    
    # Define variables based on the dataset description and plan
    Y_col = 'net_tfa'  # Net financial assets (outcome)
    W_col = 'p401'     # Participation in 401(k) (treatment)
    Z_col = 'e401'     # Eligibility for 401(k) (instrument)
    
    # Covariates (as planned)
    X_cols_original = ['inc', 'age', 'educ', 'marr', 'fsize', 'twoearn', 'db', 'pira', 'hown']
    
    required_cols = [Y_col, W_col, Z_col] + X_cols_original
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the dataset: {missing_cols}")
        return pd.DataFrame(), []

    df_subset = df[required_cols].copy()
    
    # Basic preprocessing: Handle missing values
    df_subset.dropna(inplace=True) # Simple NaN handling for this example
    
    if len(df_subset) == 0:
        print("Error: No data remaining after dropping NaNs.")
        return pd.DataFrame(), []

    # Rename columns for consistency if desired, or use original names
    # For this script, we'll use Y, W, Z and keep original X column names
    df_subset.rename(columns={
        Y_col: 'Y',
        W_col: 'W',
        Z_col: 'Z'
    }, inplace=True)
        
    return df_subset, X_cols_original

# --- 2. Nuisance Function Estimators ---
def estimate_nuisance_functions_empirical(data, X_cols, y_grid, k_folds=5):
    """
    Estimates nuisance functions using K-fold cross-fitting.
    - pi_hat(X) = P(Z=1 | X)
    - p_hat(X, Z) = P(W=1 | X, Z)
    - mu_hat_y(X, W) = E[1{Y<=y} | X, W]
    - mu_hat_y_w1(X) = E[1{Y<=y} | X, W=1]
    - mu_hat_y_w0(X) = E[1{Y<=y} | X, W=0]
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    pi_hat_oos = np.zeros(len(data))
    p_hat_oos = np.zeros(len(data))
    mu_hat_y_oos = np.zeros((len(data), len(y_grid)))
    mu_hat_y_w1_oos = np.zeros((len(data), len(y_grid)))
    mu_hat_y_w0_oos = np.zeros((len(data), len(y_grid)))

    for train_index, test_index in kf.split(data):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        # --- Feature Scaling ---
        scaler_X = StandardScaler().fit(data_train[X_cols])
        X_train_scaled = scaler_X.transform(data_train[X_cols])
        X_test_scaled = scaler_X.transform(data_test[X_cols])

        scaler_XZ = StandardScaler().fit(data_train[X_cols + ['Z']])
        XZ_train_scaled = scaler_XZ.transform(data_train[X_cols + ['Z']])
        XZ_test_scaled = scaler_XZ.transform(data_test[X_cols + ['Z']])

        scaler_XW = StandardScaler().fit(data_train[X_cols + ['W']])
        XW_train_scaled = scaler_XW.transform(data_train[X_cols + ['W']])
        XW_test_scaled = scaler_XW.transform(data_test[X_cols + ['W']])

        # Estimate pi(x) = P(Z=1 | X)
        pi_model = train_standardized_kan(X_train_scaled, data_train['Z'].values, loss_fn='bce')
        with torch.no_grad():
            pi_hat_oos[test_index] = torch.sigmoid(pi_model(torch.from_numpy(X_test_scaled).float().to(DEVICE))).cpu().numpy().flatten()

        # Estimate p(x, z) = P(W=1 | Z, X)
        p_model = train_standardized_kan(XZ_train_scaled, data_train['W'].values, loss_fn='bce')
        with torch.no_grad():
            p_hat_oos[test_index] = torch.sigmoid(p_model(torch.from_numpy(XZ_test_scaled).float().to(DEVICE))).cpu().numpy().flatten()

        for i, y_val in enumerate(y_grid):
            data_train_y_loop = data_train.copy()
            data_train_y_loop['Y_le_y'] = (data_train_y_loop['Y'] <= y_val).astype(int)
            y_le_y_train = data_train_y_loop['Y_le_y'].values

            # Estimate E[1{Y<=y} | X, W]
            if len(np.unique(y_le_y_train)) < 2 or np.min(np.bincount(y_le_y_train)) < MIN_CLASS_COUNT_THRESHOLD:
                mu_hat_y_oos[test_index, i] = np.mean(y_le_y_train)
            else:
                mu_y_model = train_standardized_kan(XW_train_scaled, y_le_y_train, loss_fn='bce')
                with torch.no_grad():
                    mu_hat_y_oos[test_index, i] = torch.sigmoid(mu_y_model(torch.from_numpy(XW_test_scaled).float().to(DEVICE))).cpu().numpy().flatten()

            # Estimate E[1{Y<=y} | X, W=1]
            data_train_y_w1 = data_train_y_loop[data_train_y_loop['W'] == 1]
            if len(data_train_y_w1) < 2 or len(data_train_y_w1['Y_le_y'].unique()) < 2 or np.min(np.bincount(data_train_y_w1['Y_le_y'])) < MIN_CLASS_COUNT_THRESHOLD:
                prop_w1_y_le_y = data_train_y_loop[data_train_y_loop['W']==1]['Y_le_y'].mean() if len(data_train_y_w1)>0 else 0.5
                mu_hat_y_w1_oos[test_index, i] = prop_w1_y_le_y if not np.isnan(prop_w1_y_le_y) else 0.5
            else:
                X_train_w1_scaled = scaler_X.transform(data_train_y_w1[X_cols])
                mu_y_w1_model = train_standardized_kan(X_train_w1_scaled, data_train_y_w1['Y_le_y'].values, loss_fn='bce')
                with torch.no_grad():
                    mu_hat_y_w1_oos[test_index, i] = torch.sigmoid(mu_y_w1_model(torch.from_numpy(X_test_scaled).float().to(DEVICE))).cpu().numpy().flatten()

            # Estimate E[1{Y<=y} | X, W=0]
            data_train_y_w0 = data_train_y_loop[data_train_y_loop['W'] == 0]
            if len(data_train_y_w0) < 2 or len(data_train_y_w0['Y_le_y'].unique()) < 2 or np.min(np.bincount(data_train_y_w0['Y_le_y'])) < MIN_CLASS_COUNT_THRESHOLD:
                prop_w0_y_le_y = data_train_y_loop[data_train_y_loop['W']==0]['Y_le_y'].mean() if len(data_train_y_w0)>0 else 0.5
                mu_hat_y_w0_oos[test_index, i] = prop_w0_y_le_y if not np.isnan(prop_w0_y_le_y) else 0.5
            else:
                X_train_w0_scaled = scaler_X.transform(data_train_y_w0[X_cols])
                mu_y_w0_model = train_standardized_kan(X_train_w0_scaled, data_train_y_w0['Y_le_y'].values, loss_fn='bce')
                with torch.no_grad():
                    mu_hat_y_w0_oos[test_index, i] = torch.sigmoid(mu_y_w0_model(torch.from_numpy(X_test_scaled).float().to(DEVICE))).cpu().numpy().flatten()
                
    nuisance_results = {
        'pi_hat': pi_hat_oos, 'p_hat': p_hat_oos,
        'mu_hat_y': mu_hat_y_oos,
        'mu_hat_y_w1': mu_hat_y_w1_oos,
        'mu_hat_y_w0': mu_hat_y_w0_oos
    }
    return nuisance_results

# --- 3. D-LATE Estimator ---
def dlate_estimator_empirical(data, nuisance_results, y_grid):
    Z = data['Z'].values
    W = data['W'].values
    Y = data['Y'].values
    
    pi_hat = nuisance_results['pi_hat']
    p_hat = nuisance_results['p_hat']
    mu_hat_y_oos = nuisance_results['mu_hat_y']       # Shape (n_samples, n_y_grid)
    mu_hat_y_w1_oos = nuisance_results['mu_hat_y_w1'] # Shape (n_samples, n_y_grid)
    mu_hat_y_w0_oos = nuisance_results['mu_hat_y_w0'] # Shape (n_samples, n_y_grid)

    epsilon = 1e-5 # Clipping value for probabilities
    pi_hat_clipped = np.clip(pi_hat, epsilon, 1 - epsilon)
    
    # Instrumental variable propensity score term
    prop_z_term = (Z - pi_hat_clipped) / (pi_hat_clipped * (1 - pi_hat_clipped))
    
    # Psi_beta: E[ ( (Z-pi(X))/(pi(X)(1-pi(X))) ) * (W - p(X,Z)) ]
    psi_beta = prop_z_term * (W - p_hat)
    mean_psi_beta = np.mean(psi_beta)
    
    if np.abs(mean_psi_beta) < epsilon: # Check if denominator is too small
        print(f"Warning: Denominator E[psi_beta] = {mean_psi_beta:.4f} is close to zero. D-LATE estimates may be unstable.")
        # Consider returning NaNs or raising an error
        # For now, we'll proceed but the results might be unreliable.
    
    dlate_estimates = []
    for i, y_val in enumerate(y_grid):
        # alpha_hat_i(y) = E[1{Y<=y} | X_i, W=1] - E[1{Y<=y} | X_i, W=0] (using OOS predictions)
        alpha_hat_y_oos = mu_hat_y_w1_oos[:, i] - mu_hat_y_w0_oos[:, i]
        
        # Psi_alpha(y): E [ prop_z_term * (1{Y<=y} - E[1{Y<=y}|X,W]) + alpha_hat_y_oos ]
        psi_alpha_term1 = prop_z_term * ((Y <= y_val).astype(int) - mu_hat_y_oos[:, i])
        psi_alpha_y = psi_alpha_term1 + alpha_hat_y_oos
        
        mean_psi_alpha_y = np.mean(psi_alpha_y)
        
        if np.abs(mean_psi_beta) < epsilon: # Avoid division by (near) zero
             dlate_y = np.nan
        else:
            dlate_y = mean_psi_alpha_y / mean_psi_beta
        dlate_estimates.append(dlate_y)
        
    return np.array(dlate_estimates)

# --- 4. Main Execution Block ---
def main():
    print("Starting D-LATE estimation for empirical application...")
    
    # Ensure results directory exists for outputs
    output_dir = "../results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Load and prepare data
    data, X_cols = load_and_prepare_data(csv_path='../data/pension.csv')
    if data.empty or len(data) < 100: # Basic check for sufficient data
        print("Data loading or preprocessing resulted in insufficient data. Exiting.")
        return

    print(f"Data loaded: {len(data)} observations after cleaning.")

    # 2. Define the grid of y values for D-LATE
    # Based on the distribution of 'Y' (net_tfa)
    min_y = np.percentile(data['Y'], 1)  # Use 1st percentile to avoid extreme outliers
    max_y = np.percentile(data['Y'], 99)  # Use 99th percentile
    if min_y >= max_y: # Handle cases where percentiles are too close or inverted
        min_y = data['Y'].min()
        max_y = data['Y'].max()
        if min_y >= max_y:
             print("Cannot determine a valid y_grid. Outcome variable might have no variance or too few unique values.")
             return

    y_grid = np.linspace(min_y, max_y, 30) # 30 points for the D-LATE curve
    print(f"y_grid for D-LATE: from {min_y:.2f} to {max_y:.2f} with {len(y_grid)} points.")
    
    # 3. Estimate nuisance functions
    print("Estimating nuisance functions (this may take a few minutes)...")
    nuisance_results = estimate_nuisance_functions_empirical(data, X_cols, y_grid, k_folds=5)
    
    # 4. Estimate D-LATE
    print("Estimating D-LATE...")
    dlate_estimates = dlate_estimator_empirical(data, nuisance_results, y_grid)
    
    # 5. Print, Save, and Plot Results
    print("\n--- D-LATE Estimation Results ---")
    results_df = pd.DataFrame({'y_value': y_grid, 'dlate_estimate': dlate_estimates})
    print(results_df)
    
    results_csv_path = os.path.join(output_dir, 'empirical_kan-d-iv-late_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nEmpirical D-LATE results saved to {results_csv_path}")

    plot_path = os.path.join(output_dir, 'empirical_kan-d-iv-late_plot.png')
    plt.figure(figsize=(10, 6))
    plt.plot(y_grid, dlate_estimates, marker='o', linestyle='-')
    plt.title('Estimated Distributional LATE (KAN-D-LATE) - Pension Data')
    plt.xlabel('Net Financial Assets (y)')
    plt.ylabel('KAN-D-LATE(y)')
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    # plt.show() # Typically not used in automated scripts

    print("\n--- End of Empirical Application ---")

if __name__ == "__main__":
    main()
