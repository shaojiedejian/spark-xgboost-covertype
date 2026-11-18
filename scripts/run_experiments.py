"""串行调度正式实验，保留每次运行的日志、模型、预测及失败证据。"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from forest_project.common import ROOT, config, sha256, write_json


def planned_runs(settings):
    configurations = {}
    for model in ["decision_tree", "random_forest", "xgboost"]:
        configurations[(model, settings["comparison_size"], 1)] = {"comparison"}
    for size in settings["train_sizes"]:
        configurations.setdefault(("xgboost", size, 1), set()).add("scale")
    for workers in [1, 2]:
        configurations.setdefault(("xgboost", settings["parallel_size"], workers), set()).add("parallel")
    return [{"model": model, "train_size": size, "workers": workers, "purposes": sorted(purposes), "repeat": repeat}
            for (model, size, workers), purposes in configurations.items()
            for repeat in range(settings["repeats"])]


def environment():
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    return {"python": sys.version, "platform": platform.platform(), "java": java.stdout + java.stderr,
            "java_home": os.environ.get("JAVA_HOME"), "python_executable": sys.executable,
            "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = config()
    directory = (args.output or ROOT / "results" / ("forest_" + time.strftime("%Y%m%d_%H%M%S"))).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    plan = planned_runs(settings)
    manifest = {"config": settings, "environment": environment(), "runs": plan,
                "dataset_manifest_sha256": sha256(ROOT / "data/processed/manifest.json"),
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                  for path in sorted((ROOT / "src").rglob("*.py"))},
                "protocol": "fixed 20% test set; same seeds; two repetitions for timing; no test-driven parameter tuning",
                "status": "running"}
    write_json(directory / "manifest.json", manifest)
    rows = []
    try:
        for index, run in enumerate(plan):
            name = f"{run['model']}_n{run['train_size']}_w{run['workers']}_r{run['repeat']}"
            run_dir = directory / name
            command = ["bash", "scripts/train.sh", "--model", run["model"], "--train-size", str(run["train_size"]),
                       "--workers", str(run["workers"]), "--run-dir", str(run_dir),
                       "--save-model", "yes" if run["repeat"] == 0 else "no"]
            run.update({"name": name, "command": command, "status": "running"})
            write_json(directory / "manifest.json", manifest)
            print(f"[{index + 1}/{len(plan)}] {name}", flush=True)
            begin = time.perf_counter()
            with (directory / (name + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            if metrics["is_smoke"] or metrics["train_size"] != run["train_size"]:
                raise AssertionError("Formal experiment unexpectedly used smoke parameters")
            row = {key: metrics[key] for key in ["model", "train_size", "test_size", "workers", "task_cpus", "local_cores",
                                                "accuracy", "macro_f1", "weighted_f1", "preprocessing_seconds",
                                                "training_seconds", "prediction_seconds", "model_reload_verified"]}
            row.update({"repeat": run["repeat"], "purposes": ";".join(run["purposes"]),
                        "run_name": name, "wall_seconds": time.perf_counter() - begin})
            rows.append(row)
            with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            run["status"] = "success"
            write_json(directory / "manifest.json", manifest)
            print(f"  accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, fit={row['training_seconds']:.2f}s", flush=True)
        manifest["status"] = "complete"
        write_json(directory / "manifest.json", manifest)
        (ROOT / "results/latest_experiment.txt").write_text(str(directory.relative_to(ROOT)) + "\n")
        print("正式实验完成", directory, flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_json(directory / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
