"""
Standardized KAN utility functions for D-IV-LATE estimation
This module provides consistent KAN implementations and training procedures
for both simulation and empirical studies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm

# Global KAN configuration parameters
KAN_STEPS = 100  # Increased from 25 for simulation
KAN_LR = 1e-3
KAN_WEIGHT_DECAY = 1e-4
KAN_REG_STRENGTH = 1e-4
KAN_HIDDEN_DIM = 32  # Increased from 16
K_FOLDS = 5  # Standardized (was 3 in simulation)
GRID_SIZE = 5
SPLINE_ORDER = 3
# Minimum count of minority class required to fit a KAN classifier reliably
MIN_CLASS_COUNT_THRESHOLD = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import appropriate KAN implementation with error handling
try:
    from efficient_kan import KAN
    print(f"Using efficient_kan library on {DEVICE}")
except ImportError:
    warnings.warn("Could not import from efficient_kan. Please install from GitHub: pip install git+https://github.com/Blealtan/efficient-kan.git")
    try:
        from kan import KAN
        print("WARNING: Using 'kan' library as a fallback.")
    except ImportError:
        raise ImportError("Neither efficient_kan nor kan are installed.")


def train_standardized_kan(X_train, y_train, X_val=None, y_val=None, verbose=False, loss_fn='mse'):
    """Standardized KAN training procedure for consistent results
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        verbose: Whether to print training progress
        
    Returns:
        Trained KAN model
    """
    n_features = X_train.shape[1]
    
    # Convert data to torch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(DEVICE)
    
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(DEVICE)
        has_validation = True
    else:
        has_validation = False
    
    # Standard architecture
    model = KAN(
        layers_hidden=[n_features, KAN_HIDDEN_DIM, 1], 
        grid_size=GRID_SIZE, 
        spline_order=SPLINE_ORDER
    ).to(DEVICE)
    
    # Standard training
    optimizer = optim.Adam(model.parameters(), lr=KAN_LR, weight_decay=KAN_WEIGHT_DECAY)
    if loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss_fn: {loss_fn}. Choose 'mse' or 'bce'.")
    
    # Initialize tracking
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Train with progress bar if verbose
    iterator = range(KAN_STEPS)
    if verbose:
        iterator = tqdm(iterator, desc="Training KAN")
    
    for step in iterator:
        # Forward pass
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        
        # Compute loss with regularization
        class_loss = criterion(y_pred, y_train_tensor)
        reg_loss = KAN_REG_STRENGTH * model.regularization_loss()
        loss = class_loss + reg_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation step
        if has_validation:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_tensor)
                # Note: Regularization is not typically included in validation loss
                val_loss = criterion(y_val_pred, y_val_tensor)
                val_losses.append(val_loss.item())
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
    
    # Use best validation model if available
    if has_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def standardized_cross_fitting(X, y, treatment_values=None):
    """Standardized k-fold cross-fitting procedure
    
    Args:
        X: Features
        y: Target variable
        treatment_values: Optional list of treatment values to fit separate models for
        
    Returns:
        Dictionary of fitted models and predictions
    """
    n_samples = len(X)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Initialize output
    models = {}
    y_pred = np.zeros_like(y)
    
    # Track folds
    fold_indices = []
    
    # Handle treatment specific case
    if treatment_values is None:
        treatment_values = [None]
    
    # Loop through treatment values if specified
    for w_val in treatment_values:
        # Get mask for this treatment value
        if w_val is not None:
            mask = (treatment_values == w_val)
            X_w = X[mask]
            y_w = y[mask]
            fold_models = []
            
            # Skip if too few samples
            if len(X_w) < 10 or len(np.unique(y_w)) <= 1:
                # Special case: constant prediction
                const_val = np.mean(y_w) if len(y_w) > 0 else 0.5
                models[w_val] = lambda x, const=const_val: np.full(len(x), const)
                y_pred[mask] = const_val
                continue
        else:
            mask = np.ones(n_samples, dtype=bool)
            X_w = X
            y_w = y
            fold_models = []
        
        # Cross-fitting
        for i, (train_idx, test_idx) in enumerate(kf.split(X_w)):
            # Store fold indices for future reference
            fold_indices.append((i, test_idx))
            
            # Get train-test split
            X_train, X_test = X_w[train_idx], X_w[test_idx]
            y_train, y_test = y_w[train_idx], y_w[test_idx]
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = train_standardized_kan(X_train_scaled, y_train)
            fold_models.append((model, scaler))
            
            # Make predictions on test fold
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
                y_test_pred = model(X_test_tensor).cpu().numpy().flatten()
            
            # Store predictions
            if w_val is not None:
                y_pred[mask][test_idx] = y_test_pred
            else:
                y_pred[test_idx] = y_test_pred
        
        # Store models
        models[w_val] = fold_models
    
    return {
        'models': models,
        'predictions': y_pred,
        'fold_indices': fold_indices
    }


def bootstrap_dlate_ci(data, y_grid, nuisance_estimator, dlate_estimator, n_bootstrap=500, alpha=0.05):
    """
    Compute bootstrap confidence intervals for D-IV-LATE estimates
    
    Args:
        data: Dataframe with variables
        y_grid: Grid of y values
        nuisance_estimator: Function to estimate nuisance functions
        dlate_estimator: Function to estimate D-IV-LATE
        n_bootstrap: Number of bootstrap replications
        alpha: Significance level
    
    Returns:
        Dictionary with confidence intervals and bootstrap estimates
    """
    n = len(data)
    bootstrap_estimates = []
    
    for b in tqdm(range(n_bootstrap), desc='Bootstrap CIs'):
        # Block bootstrap preserving cross-fitting structure
        boot_indices = np.random.choice(n, size=n, replace=True)
        boot_data = data.iloc[boot_indices].reset_index(drop=True)
        
        # Estimate nuisance functions on bootstrap sample
        nuisance_df = nuisance_estimator(boot_data, y_grid)
        
        # Compute D-IV-LATE estimates
        dlate_est = dlate_estimator(boot_data, nuisance_df, y_grid)
        bootstrap_estimates.append(dlate_est)
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Compute confidence intervals
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(bootstrap_estimates, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_estimates, upper_percentile, axis=0)
    point_est = np.mean(bootstrap_estimates, axis=0)
    
    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'point_estimates': point_est,
        'bootstrap_estimates': bootstrap_estimates
    }


def apply_multiple_testing_correction(p_values, method='fdr_bh'):
    """
    Apply multiple testing correction to p-values
    
    Args:
        p_values: Array of p-values
        method: Correction method ('fdr_bh', 'bonferroni', etc.)
        
    Returns:
        Dictionary with corrected p-values and rejection decisions
    """
    from statsmodels.stats.multitest import multipletests
    
    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=0.05, method=method
    )
    
    return {
        'rejected': rejected,
        'p_corrected': p_corrected,
        'method': method
    }

# ---------------------------------------------
#  Asymptotic inference utilities
# ---------------------------------------------

def dlate_point_se(data, nuisance_df, y_grid):
    """Compute D-IV-LATE point estimates and asymptotic standard errors.

    Args:
        data (pd.DataFrame): Must contain columns 'Z', 'W', 'Y'.
        nuisance_df (pd.DataFrame): Output from `estimate_nuisance_functions`.
        y_grid (array-like): Grid of Y values at which to evaluate the D-LATE.

    Returns:
        dict with keys:
            - dlate: np.ndarray of point estimates
            - se: np.ndarray of standard errors
            - ci_lower / ci_upper: 95% Wald CIs
            - p_values: two-sided p-values (normal approximation)
    """
    import numpy as np
    try:
        from scipy.stats import norm
        _sf = norm.sf
    except ImportError:
        # Fallback survival function using error function
        from math import erf, sqrt
        _sf = lambda x: 0.5 * (1 - erf(x / (2**0.5)))

    Z = data['Z'].values
    W = data['W'].values
    Y = data['Y'].values

    pi_hat = nuisance_df['pi_hat'].values
    p_hat_0 = nuisance_df['p_hat_0'].values
    p_hat_1 = nuisance_df['p_hat_1'].values

    # Influence function components
    psi_beta = (p_hat_1 - p_hat_0) + (Z / pi_hat) * (W - p_hat_1) - ((1 - Z) / (1 - pi_hat)) * (W - p_hat_0)
    E_psi_beta = np.mean(psi_beta)
    if np.abs(E_psi_beta) < 1e-10:
        raise ValueError("Estimated denominator (beta) is numerically zero, cannot compute D-LATE.")

    n = len(data)
    dlate, se, ci_l, ci_u, p_vals = [], [], [], [], []

    for y_val in y_grid:
        mu_hat_0 = nuisance_df[f'mu_hat_0_{y_val}'].values
        mu_hat_1 = nuisance_df[f'mu_hat_1_{y_val}'].values
        y_indicator = (Y <= y_val).astype(int)

        psi_alpha = (mu_hat_1 - mu_hat_0) + (Z / pi_hat) * (y_indicator - mu_hat_1) - ((1 - Z) / (1 - pi_hat)) * (y_indicator - mu_hat_0)
        E_psi_alpha = np.mean(psi_alpha)

        delta = E_psi_alpha / E_psi_beta
        IF = (psi_alpha - delta * psi_beta) / E_psi_beta
        var = np.var(IF, ddof=1) / n
        se_y = np.sqrt(var)

        dlate.append(delta)
        se.append(se_y)
        ci_l.append(delta - 1.96 * se_y)
        ci_u.append(delta + 1.96 * se_y)
        p_vals.append(2 * _sf(abs(delta / se_y)) if se_y > 0 else np.nan)

    return {
        'dlate': np.array(dlate),
        'se': np.array(se),
        'ci_lower': np.array(ci_l),
        'ci_upper': np.array(ci_u),
        'p_values': np.array(p_vals)
    }
