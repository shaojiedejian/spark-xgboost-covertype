"""验证真实数据、标签、训练测试隔离及不同规模的嵌套关系。"""
import json

import numpy as np
import pandas as pd

from forest_project.common import ROOT, FEATURES, config, sha256, write_json


def main():
    base = ROOT / "data/processed"
    manifest = json.loads((base / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert sha256(base / name) == expected, f"Checksum mismatch: {name}"
    frame = pd.read_parquet(base / "covertype.parquet")
    assert frame.shape == (581012, 56)
    assert len(FEATURES) == 54 and len(set(FEATURES)) == 54
    assert frame["row_id"].is_unique
    assert np.isfinite(frame[FEATURES].to_numpy()).all()
    assert set(frame["label"]) == set(range(7))
    assert (frame[[f"wilderness_{i}" for i in range(1, 5)]].sum(axis=1) == 1).all()
    assert (frame[[f"soil_{i}" for i in range(1, 41)]].sum(axis=1) == 1).all()
    test_ids = np.load(base / "test_ids.npy")
    pool_ids = np.load(base / "train_pool_ids.npy")
    assert not np.intersect1d(test_ids, pool_ids).size
    assert np.union1d(test_ids, pool_ids).size == len(frame)
    assert set(pd.read_parquet(base / "test.parquet")["row_id"]) == set(test_ids)
    previous = set()
    for size in config()["train_sizes"]:
        ids = np.load(base / f"train_{size}_ids.npy")
        subset = pd.read_parquet(base / f"train_{size}.parquet")
        assert len(ids) == len(subset) == size
        assert len(np.unique(ids)) == size
        assert previous <= set(ids)
        assert not np.intersect1d(ids, test_ids).size
        assert set(subset["row_id"]) == set(ids)
        expected = frame.set_index("row_id").loc[subset["row_id"]].reset_index()
        pd.testing.assert_frame_equal(subset.reset_index(drop=True), expected, check_dtype=True)
        previous = set(ids)
    checks = ["official dataset dimensions and finite values", "all seven labels", "one-hot wilderness and soil invariants",
              "train/test disjointness and complete coverage", "nested training subsets with exact expected data", "all file checksums"]
    write_json(ROOT / "results/data_verification.json", {"passed": checks, "test_size": len(test_ids)})
    print("数据验证通过", len(checks), "项")


if __name__ == "__main__":
    main()
