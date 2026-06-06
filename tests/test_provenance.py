import importlib.util
from pathlib import Path

from provenance_assertions import assert_manifest_provenance

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "kan-d-iv-late" / "code"


def load_provenance_module():
    path = CODE_DIR / "provenance.py"
    spec = importlib.util.spec_from_file_location("provenance_schema_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_run_provenance_returns_manifest_safe_schema():
    provenance = load_provenance_module().collect_run_provenance(project_root=ROOT)

    assert_manifest_provenance({"provenance": provenance})
