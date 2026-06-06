"""Shared experiment configuration helpers for D-IV-LATE runners."""

from __future__ import annotations

from .kan_utils import build_kan_config, build_kan_config_id

RF_CONFIG_ID = "rf_core_v1"
KAN_CONFIG_ID = "kan_core_v1"

KAN_ABLATION_CONFIG_LIBRARY = (
    {
        "label": "kan_core_v1",
        "params": build_kan_config(steps=25, hidden_dim=16, grid_size=4, reg_strength=1e-4),
    },
    {
        "label": "kan_width8_v1",
        "params": build_kan_config(steps=25, hidden_dim=8, grid_size=4, reg_strength=1e-4),
    },
    {
        "label": "kan_width64_v1",
        "params": build_kan_config(steps=25, hidden_dim=64, grid_size=4, reg_strength=1e-4),
    },
    {
        "label": "kan_steps10_v1",
        "params": build_kan_config(steps=10, hidden_dim=16, grid_size=4, reg_strength=1e-4),
    },
    {
        "label": "kan_steps50_v1",
        "params": build_kan_config(steps=50, hidden_dim=16, grid_size=4, reg_strength=1e-4),
    },
    {
        "label": "kan_reg1e-5_v1",
        "params": build_kan_config(steps=25, hidden_dim=16, grid_size=4, reg_strength=1e-5),
    },
    {
        "label": "kan_reg1e-3_v1",
        "params": build_kan_config(steps=25, hidden_dim=16, grid_size=4, reg_strength=1e-3),
    },
    {
        "label": "kan_grid3_v1",
        "params": build_kan_config(steps=25, hidden_dim=16, grid_size=3, reg_strength=1e-4),
    },
    {
        "label": "kan_grid6_v1",
        "params": build_kan_config(steps=25, hidden_dim=16, grid_size=6, reg_strength=1e-4),
    },
)

KAN_CONFIG_LABELS = tuple(item["label"] for item in KAN_ABLATION_CONFIG_LIBRARY)


def get_labeled_kan_config(label: str) -> dict:
    """Return the KAN hyperparameter dictionary for a named policy."""
    for item in KAN_ABLATION_CONFIG_LIBRARY:
        if item["label"] == label:
            return dict(item["params"])
    raise ValueError(f"Unsupported KAN config label: {label}")


def get_labeled_kan_record(label: str) -> dict:
    """Return the manifest-ready KAN policy record for a named policy."""
    config = get_labeled_kan_config(label)
    return {
        "kan_config_label": label,
        "kan_config": config,
        "kan_config_id": build_kan_config_id(config),
        "kan_steps": config["steps"],
    }
