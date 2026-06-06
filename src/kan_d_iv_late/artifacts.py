"""Artifact writing helpers shared by runners.

These helpers intentionally preserve the existing file formats: CSV files are
written without an index, and JSON manifests are sorted and indented.
"""

from __future__ import annotations

import json
from pathlib import Path


def ensure_directory(path) -> Path:
    """Create and return an artifact directory path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_csv(dataframe, path) -> Path:
    """Write a dataframe to CSV using the repository artifact convention."""
    output_path = Path(path)
    dataframe.to_csv(output_path, index=False)
    return output_path


def write_json(payload, path) -> Path:
    """Write a manifest JSON file using the repository artifact convention."""
    output_path = Path(path)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def read_json(path):
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
