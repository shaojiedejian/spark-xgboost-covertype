"""核对已加载模型的全测试集预测及无标签 CSV 的正常预测入口。"""
import json

import numpy as np
import pandas as pd

from forest_project.common import ROOT, config, write_json


def read_spark_csv(directory):
    files = sorted((directory / "predictions").glob("part-*.csv"))
    if not files:
        raise AssertionError("Independent prediction CSV files are missing")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True).sort_values("row_id").reset_index(drop=True)


def main():
    directory = ROOT / (ROOT / "results/latest_experiment.txt").read_text().strip()
    size = max(config()["train_sizes"])
    expected = pd.read_parquet(directory / f"xgboost_n{size}_w1_r0/predictions.parquet").sort_values("row_id").reset_index(drop=True)
    actual = read_spark_csv(directory / "independent_prediction")
    pd.testing.assert_frame_equal(actual[['row_id', 'label', 'prediction']],
                                  expected[['row_id', 'label', 'prediction']], check_dtype=False)
    np.testing.assert_array_equal(actual["predicted_cover_type"], actual["prediction"].astype(int) + 1)
    assert actual["predicted_class_name"].notna().all()
    unlabeled = read_spark_csv(directory / "unlabeled_prediction")
    assert "label" not in unlabeled.columns
    sample_ids = pd.read_csv(ROOT / "data/demo/unlabeled.csv")["row_id"]
    sample = expected[expected["row_id"].isin(sample_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(unlabeled[['row_id', 'prediction']], sample[['row_id', 'prediction']], check_dtype=False)
    write_json(directory / "prediction_audit.json", {"passed": True, "full_test_rows": len(actual),
                                                       "unlabeled_rows": len(unlabeled),
                                                       "full_predictions_match_training_evaluation": True,
                                                       "unlabeled_input_verified": True})
    print("独立预测验证通过", len(actual), "条有标签数据及", len(unlabeled), "条无标签数据")


if __name__ == "__main__":
    main()
