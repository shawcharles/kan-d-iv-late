import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "kan-d-iv-late" / "code"
SIBLING_EFFICIENT_KAN_SRC = Path("/home/user/Documents/GITHUB/shawcharles/efficient-kan/src")


def test_real_sibling_efficient_kan_supports_active_kan_utils():
    if not SIBLING_EFFICIENT_KAN_SRC.exists():
        pytest.skip(f"Sibling efficient-kan checkout not found: {SIBLING_EFFICIENT_KAN_SRC}")

    env = os.environ.copy()
    pythonpath_entries = [
        str(SIBLING_EFFICIENT_KAN_SRC),
        str(CODE_DIR),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(entry for entry in pythonpath_entries if entry)

    script = """
import numpy as np
from kan_utils import build_binary_kan, predict_binary_kan

model = build_binary_kan(2, hidden_dim=2, grid_size=2, spline_order=1)
predictions = predict_binary_kan(model, np.zeros((3, 2), dtype=np.float32))
assert predictions.shape == (3,)
assert np.isfinite(predictions).all()
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
