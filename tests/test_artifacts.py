import pandas as pd

from kan_d_iv_late.artifacts import ensure_directory, read_json, write_csv, write_json


def test_artifact_helpers_preserve_csv_and_json_conventions(tmp_path):
    artifact_dir = ensure_directory(tmp_path / "nested" / "artifacts")
    assert artifact_dir.exists()

    csv_path = artifact_dir / "example.csv"
    write_csv(pd.DataFrame({"value": [1, 2]}), csv_path)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "value"

    json_path = artifact_dir / "manifest.json"
    payload = {"z": 1, "a": {"nested": True}}
    write_json(payload, json_path)
    assert read_json(json_path) == payload
    assert json_path.read_text(encoding="utf-8").startswith('{\n  "a"')
