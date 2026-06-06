"""Shared KAN helpers for the active D-IV-LATE code path."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier

try:
    from efficient_kan import KAN
except ImportError as exc:  # pragma: no cover - exercised by runtime environment
    raise ImportError(
        "efficient_kan is required for the active kan-d-iv-late code path. "
        "Install it before running the empirical or simulation scripts."
    ) from exc


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAN_DEFAULT_STEPS = 25
KAN_LR = 1e-3
KAN_WEIGHT_DECAY = 1e-4
KAN_REG_STRENGTH = 1e-4
KAN_HIDDEN_DIM = 16
GRID_SIZE = 4
SPLINE_ORDER = 3
MIN_CLASS_COUNT_THRESHOLD = 5
DEFAULT_PROBABILITY_EPSILON = 1e-6
RF_DEFAULT_N_ESTIMATORS = 100
RF_DEFAULT_MIN_SAMPLES_LEAF = 1


def clip_probabilities(values, epsilon=DEFAULT_PROBABILITY_EPSILON):
    """Clip probabilities away from 0 and 1 for stable downstream use."""
    return np.clip(np.asarray(values, dtype=float), epsilon, 1 - epsilon)


def build_kan_config(
    *,
    steps=KAN_DEFAULT_STEPS,
    hidden_dim=KAN_HIDDEN_DIM,
    grid_size=GRID_SIZE,
    spline_order=SPLINE_ORDER,
    lr=KAN_LR,
    weight_decay=KAN_WEIGHT_DECAY,
    reg_strength=KAN_REG_STRENGTH,
    min_class_count=MIN_CLASS_COUNT_THRESHOLD,
    probability_epsilon=DEFAULT_PROBABILITY_EPSILON,
):
    """Return a normalized KAN hyperparameter dictionary."""
    return {
        "steps": int(steps),
        "hidden_dim": int(hidden_dim),
        "grid_size": int(grid_size),
        "spline_order": int(spline_order),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "reg_strength": float(reg_strength),
        "min_class_count": int(min_class_count),
        "probability_epsilon": float(probability_epsilon),
    }


def default_kan_config():
    """Return the canonical active-tree KAN configuration."""
    return build_kan_config()


def build_kan_config_id(config, *, default_id="kan_core_v1"):
    """Build a stable manifest-friendly identifier for a KAN configuration."""
    normalized = build_kan_config(**config)
    if normalized == default_kan_config():
        return default_id

    reg_component = f"{normalized['reg_strength']:.0e}".replace("+0", "").replace("+", "")
    lr_component = f"{normalized['lr']:.0e}".replace("+0", "").replace("+", "")
    wd_component = f"{normalized['weight_decay']:.0e}".replace("+0", "").replace("+", "")
    return (
        "kan"
        f"_hd{normalized['hidden_dim']}"
        f"_gs{normalized['grid_size']}"
        f"_sp{normalized['spline_order']}"
        f"_st{normalized['steps']}"
        f"_lr{lr_component}"
        f"_wd{wd_component}"
        f"_reg{reg_component}"
    )


def build_rf_config(
    *,
    n_estimators=RF_DEFAULT_N_ESTIMATORS,
    min_samples_leaf=RF_DEFAULT_MIN_SAMPLES_LEAF,
    random_state=42,
):
    """Return a normalized Random Forest hyperparameter dictionary."""
    return {
        "n_estimators": int(n_estimators),
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
    }


def default_rf_config():
    """Return the canonical active-tree RF configuration."""
    return build_rf_config()


def build_rf_config_id(config, *, default_id="rf_core_v1"):
    """Build a stable manifest-friendly identifier for an RF configuration."""
    normalized = build_rf_config(**config)
    if normalized == default_rf_config():
        return default_id
    return f"rf_nt{normalized['n_estimators']}_leaf{normalized['min_samples_leaf']}"


def _to_float_tensor(values):
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=DEVICE)


def build_binary_kan(
    input_dim,
    *,
    hidden_dim=KAN_HIDDEN_DIM,
    grid_size=GRID_SIZE,
    spline_order=SPLINE_ORDER,
):
    """Construct the active-tree binary KAN architecture."""
    return KAN(
        layers_hidden=[input_dim, hidden_dim, 1],
        grid_size=grid_size,
        spline_order=spline_order,
    ).to(DEVICE)


def train_binary_kan(
    model,
    train_features,
    train_labels,
    *,
    steps=KAN_DEFAULT_STEPS,
    lr=KAN_LR,
    weight_decay=KAN_WEIGHT_DECAY,
    reg_strength=KAN_REG_STRENGTH,
):
    """Train a binary KAN with BCE loss and spline regularization."""
    train_inputs = _to_float_tensor(train_features)
    train_targets = _to_float_tensor(np.asarray(train_labels).reshape(-1, 1))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(train_inputs)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        loss = criterion(logits, train_targets)
        loss = loss + reg_strength * model.regularization_loss()
        loss.backward()
        optimizer.step()

    return model


def predict_binary_kan(model, features):
    """Predict binary probabilities with a trained KAN."""
    test_inputs = _to_float_tensor(features)
    model.eval()
    with torch.no_grad():
        logits = model(test_inputs)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    return probabilities


def fit_binary_kan_predict(
    train_features,
    train_labels,
    test_features,
    *,
    steps=KAN_DEFAULT_STEPS,
    hidden_dim=KAN_HIDDEN_DIM,
    grid_size=GRID_SIZE,
    spline_order=SPLINE_ORDER,
    lr=KAN_LR,
    weight_decay=KAN_WEIGHT_DECAY,
    reg_strength=KAN_REG_STRENGTH,
    min_class_count=MIN_CLASS_COUNT_THRESHOLD,
    fallback_probability=0.5,
    clip=False,
    epsilon=DEFAULT_PROBABILITY_EPSILON,
):
    """
    Fit a binary KAN and return test-set probabilities.

    If the training labels are empty, constant, or too imbalanced for a stable
    classifier fit, the function returns a constant fallback prediction instead.
    """
    test_features = np.asarray(test_features, dtype=np.float32)
    if test_features.shape[0] == 0:
        return np.array([], dtype=float)

    train_features = np.asarray(train_features, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.float32).reshape(-1)

    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features and train_labels must have the same length")

    if train_labels.size == 0:
        predictions = np.full(test_features.shape[0], fallback_probability, dtype=float)
    else:
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        if len(unique_labels) == 1:
            predictions = np.full(test_features.shape[0], float(unique_labels[0]), dtype=float)
        elif np.min(counts) < min_class_count:
            majority_label = float(unique_labels[np.argmax(counts)])
            predictions = np.full(test_features.shape[0], majority_label, dtype=float)
        else:
            model = build_binary_kan(
                train_features.shape[1],
                hidden_dim=hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
            )
            train_binary_kan(
                model,
                train_features,
                train_labels,
                steps=steps,
                lr=lr,
                weight_decay=weight_decay,
                reg_strength=reg_strength,
            )
            predictions = predict_binary_kan(model, test_features)

    if clip:
        return clip_probabilities(predictions, epsilon=epsilon)
    return predictions


def fit_binary_rf_predict(
    train_features,
    train_labels,
    test_features,
    *,
    n_estimators=RF_DEFAULT_N_ESTIMATORS,
    min_samples_leaf=RF_DEFAULT_MIN_SAMPLES_LEAF,
    random_state=42,
    fallback_probability=0.5,
    clip=False,
    epsilon=DEFAULT_PROBABILITY_EPSILON,
):
    """Fit a binary RF classifier and return test-set probabilities."""
    test_features = np.asarray(test_features)
    if test_features.shape[0] == 0:
        return np.array([], dtype=float)

    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels).reshape(-1)

    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features and train_labels must have the same length")

    if train_labels.size == 0:
        predictions = np.full(test_features.shape[0], fallback_probability, dtype=float)
    else:
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        if len(unique_labels) == 1:
            predictions = np.full(test_features.shape[0], float(unique_labels[0]), dtype=float)
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            model.fit(train_features, train_labels)
            predictions = model.predict_proba(test_features)[:, 1]

    if clip:
        return clip_probabilities(predictions, epsilon=epsilon)
    return predictions
