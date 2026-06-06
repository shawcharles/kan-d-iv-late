from pathlib import Path

from kan_d_iv_late.provenance import collect_run_provenance
from provenance_assertions import assert_manifest_provenance

ROOT = Path(__file__).resolve().parents[1]


def test_collect_run_provenance_returns_manifest_safe_schema():
    provenance = collect_run_provenance(project_root=ROOT)

    assert_manifest_provenance({"provenance": provenance})
