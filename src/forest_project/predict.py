"""加载已保存的 Spark Pipeline，对独立文件执行批量预测。"""
import argparse
from pathlib import Path
import time

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from .common import FEATURES, CLASS_NAMES, config, write_json
from .train import start_spark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "spark-events").mkdir()
    spark = None
    try:
        spark, _ = start_spark(config(), "inference", 1, output)
        if args.input.suffix == ".parquet":
            data = spark.read.parquet(str(args.input.resolve()))
        elif args.input.suffix == ".csv":
            data = spark.read.option("header", True).option("inferSchema", True).csv(str(args.input.resolve()))
        else:
            raise ValueError("Input must be a Parquet file or CSV with feature-name header")
        missing = set(FEATURES) - set(data.columns)
        if missing:
            raise ValueError("Missing features: " + ", ".join(sorted(missing)))
        if "row_id" not in data.columns:
            data = data.withColumn("row_id", F.monotonically_increasing_id())
        model = PipelineModel.load(str(args.model_dir.resolve()))
        started = time.perf_counter()
        predicted = model.transform(data)
        mapping = F.create_map(*[value for pair in enumerate(CLASS_NAMES) for value in (F.lit(pair[0]), F.lit(pair[1]))])
        columns = ["row_id", "prediction"] + (["label"] if "label" in data.columns else [])
        selected = predicted.select(*columns).withColumn("predicted_cover_type", F.col("prediction").cast("int") + 1)
        selected = selected.withColumn("predicted_class_name", mapping[F.col("prediction").cast("int")]).cache()
        count = selected.count()
        selected.write.option("header", True).csv(str(output / "predictions"))
        write_json(output / "metadata.json", {"rows": count, "prediction_seconds": time.perf_counter() - started,
                                             "model": str(args.model_dir.resolve()), "input": str(args.input.resolve()),
                                             "label_convention": "prediction 0..6; predicted_cover_type 1..7"})
        selected.unpersist()
        print("预测完成", output, count, "rows")
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
