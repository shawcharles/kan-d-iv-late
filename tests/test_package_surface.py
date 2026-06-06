import pandas as pd

from kan_d_iv_late.config import get_labeled_kan_record
from kan_d_iv_late.crossfit import crossfit_splits
from kan_d_iv_late.dlate_inference import bootstrap_dlate_inference
from kan_d_iv_late.dlate_score import compute_dlate_score_objects


def test_package_exports_core_surfaces():
    assert callable(compute_dlate_score_objects)
    assert callable(bootstrap_dlate_inference)


def test_shared_labeled_kan_record_is_manifest_ready():
    record = get_labeled_kan_record("kan_width64_v1")

    assert record["kan_config_label"] == "kan_width64_v1"
    assert record["kan_config"]["hidden_dim"] == 64
    assert record["kan_steps"] == 25
    assert record["kan_config_id"].startswith("kan_hd64")


def test_crossfit_splits_are_deterministic_and_complete():
    data = pd.DataFrame({"x": range(12)})

    first = list(crossfit_splits(data, n_splits=3))
    second = list(crossfit_splits(data, n_splits=3))

    assert len(first) == 3
    assert [test.tolist() for _, test in first] == [test.tolist() for _, test in second]
    assert sorted(index for _, test in first for index in test.tolist()) == list(range(12))
