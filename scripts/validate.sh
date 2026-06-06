#!/usr/bin/env bash
set -euo pipefail

python -m compileall src/kan_d_iv_late kan-d-iv-late/code tests
pytest -q tests
git diff --check
