"""从每行实际预测重算指标，审计全部正式实验而非仅检查退出码。"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from forest_project.common import ROOT, CLASS_NAMES, config, sha256, write_json


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    with (directory / "summary.csv").open() as source:
        summary = list(csv.DictReader(source))
    settings = manifest["config"]
    combinations = {(model, settings["comparison_size"], 1) for model in ["decision_tree", "random_forest", "xgboost"]}
    combinations.update(("xgboost", size, 1) for size in settings["train_sizes"])
    combinations.update(("xgboost", settings["parallel_size"], workers) for workers in [1, 2])
    count = len(combinations) * settings["repeats"]
    assert len(summary) == len(manifest["runs"]) == count
    assert len({row["run_name"] for row in summary}) == count
    expected = pd.read_parquet(ROOT / "data/processed/test.parquet")[['row_id', 'label']].sort_values("row_id").reset_index(drop=True)
    ids_hash = sha256(ROOT / "data/processed/test_ids.npy")
    reports = []
    for run, row in zip(manifest["runs"], summary):
        assert run["status"] == "success" and run["name"] == row["run_name"]
        folder = directory / run["name"]
        actual = pd.read_parquet(folder / "predictions.parquet").sort_values("row_id").reset_index(drop=True)
        assert len(actual) == len(expected)
        pd.testing.assert_frame_equal(actual[['row_id', 'label']], expected, check_dtype=False)
        predicted = actual["prediction"].to_numpy(dtype=int)
        assert np.isin(predicted, np.arange(7)).all()
        metrics = json.loads((folder / "metrics.json").read_text())
        assert not metrics["is_smoke"] and metrics["train_size"] == run["train_size"]
        assert metrics["test_ids_sha256"] == ids_hash
        assert metrics["test_parquet_sha256"] == sha256(ROOT / "data/processed/test.parquet")
        for metric, recomputed in [
            ("accuracy", accuracy_score(expected["label"], predicted)),
            ("macro_f1", f1_score(expected["label"], predicted, average="macro", zero_division=0)),
            ("weighted_f1", f1_score(expected["label"], predicted, average="weighted", zero_division=0)),
        ]:
            assert math.isclose(metrics[metric], recomputed, abs_tol=1e-12)
            assert math.isclose(float(row[metric]), recomputed, abs_tol=1e-12)
        matrix = pd.read_csv(folder / "confusion_matrix.csv", index_col=0).to_numpy()
        np.testing.assert_array_equal(matrix, confusion_matrix(expected["label"], predicted, labels=np.arange(7)))
        for metric in ["preprocessing_seconds", "training_seconds", "prediction_seconds"]:
            assert math.isfinite(metrics[metric]) and metrics[metric] > 0
        if run["repeat"] == 0:
            assert metrics["model_reload_verified"] and (folder / "model").is_dir()
        if run["model"] == "xgboost":
            assert metrics["parameters"]["device"] == "cpu"
            assert metrics["workers"] * metrics["task_cpus"] == config()["local_cores"]
            log = (directory / (run["name"] + ".log")).read_text()
            assert "Training on CPUs" in log
            assert f"'num_class': 7" in log
            assert f"'nthread': {metrics['task_cpus']}" in log
            assert metrics["workers"] == run["workers"]
            assert metrics["task_cpus"] * metrics["workers"] == metrics["local_cores"]
            if run["workers"] == 2:
                assert "Rank 1" in log or "rank 1" in log
        reports.append({"run": run["name"], "rows_verified": len(actual), "metrics_recomputed": True})
    write_json(directory / "audit.json", {"passed": True, "runs": reports,
                                           "scope": "all predictions, fixed test IDs and labels, metrics, confusion matrices, CPU logs and model reload evidence"})
    print("完整结果审计通过", len(reports), "runs")


if __name__ == "__main__":
    verify(ROOT / (ROOT / "results/latest_experiment.txt").read_text().strip())
