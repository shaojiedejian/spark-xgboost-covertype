"""真实 PySpark 模型训练、CPU 分布式 XGBoost、评价和模型保存。"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, functions as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost.spark import SparkXGBClassifier

from .common import ROOT, FEATURES, CLASS_NAMES, config, sha256, write_json


def start_spark(settings, model, workers, directory):
    cores = settings["local_cores"]
    task_cpus = cores // workers if model == "xgboost" else 1
    if workers < 1 or workers > cores or cores % workers:
        raise ValueError("Worker count must divide the available local cores")
    builder = (SparkSession.builder.master(f"local[{cores}]")
               .appName(f"Covertype-{model}-{workers}workers")
               .config("spark.driver.memory", settings["driver_memory"])
               .config("spark.driver.bindAddress", "127.0.0.1")
               .config("spark.driver.host", "127.0.0.1")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", settings["partitions"])
               .config("spark.default.parallelism", settings["partitions"])
               .config("spark.task.cpus", task_cpus)
               .config("spark.sql.execution.arrow.pyspark.enabled", "true")
               .config("spark.python.worker.reuse", "true")
               .config("spark.eventLog.enabled", "true")
               .config("spark.eventLog.dir", (directory / "spark-events").as_uri()))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark, task_cpus


def estimator_for(args, settings):
    shared = {"featuresCol": "features", "labelCol": "label", "seed": settings["seed"]}
    if args.model == "decision_tree":
        params = dict(settings["decision_tree"])
        return DecisionTreeClassifier(**shared, **params), params
    if args.model == "random_forest":
        params = dict(settings["random_forest"])
        if args.trees:
            params["numTrees"] = args.trees
        return RandomForestClassifier(**shared, **params), params
    params = dict(settings["xgboost"])
    if args.trees:
        params["n_estimators"] = args.trees
    model = SparkXGBClassifier(features_col="features", label_col="label",
                               num_workers=args.workers, random_state=settings["seed"],
                               eval_metric="mlogloss", force_repartition=True,
                               **params)
    return model, params


def select_rows(frame, limit):
    if limit:
        # 调试入口固定排序后截取，批量正式实验不使用该参数。
        return frame.orderBy("row_id").limit(limit)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["decision_tree", "random_forest", "xgboost"], required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--trees", type=int)
    parser.add_argument("--save-model", choices=["yes", "no"], default="yes")
    args = parser.parse_args()
    for name in ["train_size", "workers", "train_limit", "test_limit", "trees"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    settings = config()
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "spark-events").mkdir()
    dataset = ROOT / "data/processed"
    train_path = dataset / f"train_{args.train_size}.parquet"
    test_path = dataset / "test.parquet"
    started = time.perf_counter()
    spark = None
    try:
        spark, task_cpus = start_spark(settings, args.model, args.workers, directory)
        preparation = time.perf_counter()
        assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="error")
        raw_train = select_rows(spark.read.parquet(str(train_path)), args.train_limit)
        raw_test = select_rows(spark.read.parquet(str(test_path)), args.test_limit)
        train = (assembler.transform(raw_train).select("row_id", "label", "features")
                 .repartition(settings["partitions"], "row_id")
                 .persist(StorageLevel.MEMORY_AND_DISK))
        test = (assembler.transform(raw_test).select("row_id", "label", "features")
                .repartition(settings["partitions"], "row_id")
                .persist(StorageLevel.MEMORY_AND_DISK))
        train_count, test_count = train.count(), test.count()
        invalid = train.filter(F.col("label").isNull() | ~F.col("label").isin(list(range(7)))).count()
        if invalid or train.select("label").distinct().count() != 7:
            raise ValueError("Training labels must contain all seven valid classes")
        preprocessing_seconds = time.perf_counter() - preparation
        estimator, params = estimator_for(args, settings)
        fit_started = time.perf_counter()
        model = estimator.fit(train)
        training_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = model.transform(test).select("row_id", "label", "prediction").toPandas()
        prediction_seconds = time.perf_counter() - predict_started
        predictions = predictions.sort_values("row_id").reset_index(drop=True)
        if len(predictions) != test_count or predictions["row_id"].duplicated().any():
            raise AssertionError("Missing or duplicate test predictions")
        truth = predictions["label"].to_numpy(dtype=int)
        predicted = predictions["prediction"].to_numpy(dtype=int)
        if not np.isin(predicted, np.arange(7)).all():
            raise AssertionError("Predicted labels outside 0..6")
        predictions.to_parquet(directory / "predictions.parquet", index=False)
        matrix = confusion_matrix(truth, predicted, labels=np.arange(7))
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(directory / "confusion_matrix.csv")
        report = classification_report(truth, predicted, labels=np.arange(7), target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        write_json(directory / "classification_report.json", report)
        if args.model == "xgboost":
            gains = model.get_booster().get_score(importance_type="gain")
            importance = np.array([gains.get(f"f{index}", gains.get(name, 0.0))
                                   for index, name in enumerate(FEATURES)])
            if importance.sum():
                importance /= importance.sum()
        else:
            importance = model.featureImportances.toArray()
        pd.DataFrame({"feature": FEATURES, "importance": importance}).sort_values(
            "importance", ascending=False).to_csv(directory / "feature_importance.csv", index=False)

        persistence_seconds = 0.0
        if args.save_model == "yes":
            save_started = time.perf_counter()
            pipeline = PipelineModel(stages=[assembler, model])
            pipeline.write().save(str(directory / "model"))
            if args.model == "xgboost":
                model.get_booster().save_model(str(directory / "booster.json"))
            # 经正常加载入口重新预测同一批固定样本，检查模型交付可用性。
            sample = raw_test.orderBy("row_id").limit(256).cache()
            before = pipeline.transform(sample).select("row_id", "prediction").orderBy("row_id").collect()
            restored = PipelineModel.load(str(directory / "model"))
            after = restored.transform(sample).select("row_id", "prediction").orderBy("row_id").collect()
            if before != after:
                raise AssertionError("Saved/reloaded model predictions differ")
            sample.unpersist()
            persistence_seconds = time.perf_counter() - save_started

        metrics = {
            "model": args.model, "train_size": train_count, "test_size": test_count,
            "workers": args.workers if args.model == "xgboost" else None,
            "local_cores": settings["local_cores"], "task_cpus": task_cpus,
            "preprocessing_seconds": preprocessing_seconds, "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds, "persistence_check_seconds": persistence_seconds,
            "accuracy": accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth, predicted, average="weighted", zero_division=0),
            "parameters": params, "train_parquet_sha256": sha256(train_path),
            "test_parquet_sha256": sha256(test_path),
            "test_ids_sha256": sha256(dataset / "test_ids.npy"),
            "model_reload_verified": args.save_model == "yes",
            "spark_version": spark.version, "platform": platform.platform(),
            "is_smoke": bool(args.train_limit or args.test_limit or args.trees),
        }
        train.unpersist()
        test.unpersist()
        metrics["elapsed_before_spark_stop_seconds"] = time.perf_counter() - started
        write_json(directory / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
