"""Shared cross-fitting helpers."""

from __future__ import annotations

from sklearn.model_selection import KFold


def crossfit_splits(data, *, n_splits: int, random_state: int = 42):
    """Yield deterministic train/test indices for cross-fitted nuisance estimation."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    yield from splitter.split(data)
