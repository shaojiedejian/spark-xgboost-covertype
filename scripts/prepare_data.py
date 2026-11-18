"""下载并校验 Covertype，固定测试集，保存可审计的嵌套训练子集。"""
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from forest_project.common import ROOT, FEATURES, CLASS_NAMES, config, sha256, write_json

ZIP_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
RAW_CHECKSUM = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"


def acquire_raw():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "covertype.zip"
    downloaded = raw / "covertype.zip.part"
    if not archive.exists():
        if downloaded.exists():
            # 上次中断留下的文件必须通过完整 ZIP CRC 检查才能复用。
            try:
                with zipfile.ZipFile(downloaded) as source:
                    if source.testzip() is not None:
                        raise ValueError("ZIP CRC failed")
                downloaded.replace(archive)
            except (zipfile.BadZipFile, ValueError):
                downloaded.unlink()
        if not archive.exists():
            subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "15", "--max-time", "240",
                            ZIP_URL, "-o", str(downloaded)], check=True)
            with zipfile.ZipFile(downloaded) as source:
                if source.testzip() is not None:
                    raise ValueError("Downloaded ZIP failed CRC check")
            downloaded.replace(archive)
    with zipfile.ZipFile(archive) as source:
        for name in ["covtype.data.gz", "covtype.info"]:
            matches = [item for item in source.namelist() if Path(item).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member {name}")
            destination = raw / name
            if not destination.exists():
                destination.write_bytes(source.read(matches[0]))
    source = raw / "covtype.data.gz"
    if sha256(source) != RAW_CHECKSUM:
        raise ValueError("Raw data differs from the published scikit-learn Covertype checksum")
    return source


def nested_subsets(train_ids, labels, sizes, seed):
    if sizes != sorted(set(sizes)) or sizes[0] < 7 or sizes[-1] > len(train_ids):
        raise ValueError("Training sizes must be sorted, unique and fit the training pool")
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(train_ids[labels[train_ids] == label]) for label in range(7)]
    proportions = np.array([len(pool) for pool in pools]) / len(train_ids)
    subsets = {}
    for size in sizes:
        target = proportions * size
        counts = np.floor(target).astype(int)
        order = np.argsort(-(target - counts), kind="stable")
        counts[order[:size - counts.sum()]] += 1
        ids = np.sort(np.concatenate([pool[:count] for pool, count in zip(pools, counts)]))
        if len(ids) != size or len(np.unique(ids)) != size:
            raise AssertionError("Incorrect training subset size")
        subsets[size] = ids
    for small, large in zip(sizes, sizes[1:]):
        if not np.isin(subsets[small], subsets[large]).all():
            raise AssertionError("Training subsets must be nested")
    return subsets


def main():
    begin = time.perf_counter()
    settings = config()
    source = acquire_raw()
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    # 元数据与文件校验值一致时，直接复用已有数据。
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        valid = (previous.get("raw_sha256") == sha256(source)
                 and previous.get("train_sizes") == settings["train_sizes"]
                 and previous.get("seed") == settings["seed"]
                 and previous.get("test_fraction") == settings["test_fraction"]
                 and all((output / name).exists() and sha256(output / name) == value
                         for name, value in previous["sha256"].items()))
        if valid:
            print("复用已校验的 Covertype 数据和固定划分。")
            return
    frame = pd.read_csv(source, header=None, dtype=np.int32)
    if frame.shape != (581012, 55) or frame.isna().any().any():
        raise ValueError(f"Unexpected Covertype shape or missing data: {frame.shape}")
    labels = frame.iloc[:, -1].to_numpy() - 1
    if not np.array_equal(np.unique(labels), np.arange(7)):
        raise ValueError("Expected all seven labels")
    binary = frame.iloc[:, 10:54].to_numpy()
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("Expected binary wilderness and soil indicators")
    if not (binary[:, :4].sum(axis=1) == 1).all() or not (binary[:, 4:].sum(axis=1) == 1).all():
        raise ValueError("Expected one wilderness and one soil category per observation")
    points = frame.iloc[:, :54].astype(np.float32)
    points.columns = FEATURES
    points.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    points["label"] = labels.astype(np.float64)
    train_ids, test_ids = train_test_split(np.arange(len(frame)), test_size=settings["test_fraction"],
                                         stratify=labels, random_state=settings["seed"])
    train_ids, test_ids = np.sort(train_ids), np.sort(test_ids)
    subsets = nested_subsets(train_ids, labels, settings["train_sizes"], settings["seed"])
    if np.intersect1d(train_ids, test_ids).size:
        raise AssertionError("Train/test leakage")
    points.to_parquet(output / "covertype.parquet", index=False)
    points.iloc[test_ids].to_parquet(output / "test.parquet", index=False)
    np.save(output / "test_ids.npy", test_ids)
    np.save(output / "train_pool_ids.npy", train_ids)
    for size, ids in subsets.items():
        points.iloc[ids].to_parquet(output / f"train_{size}.parquet", index=False)
        np.save(output / f"train_{size}_ids.npy", ids)
    files = [path for path in output.iterdir() if path.suffix in [".npy", ".parquet"]]
    description = {
        "dataset": "UCI Covertype", "doi": "10.24432/C50K5N", "download_url": ZIP_URL,
        "n_rows": len(frame), "n_features": len(FEATURES), "test_size": len(test_ids),
        "train_pool_size": len(train_ids), "train_sizes": settings["train_sizes"],
        "seed": settings["seed"], "test_fraction": settings["test_fraction"],
        "split": "stratified row split; nested stratified training subsets",
        "label_conversion": "original 1..7 to 0..6", "features": FEATURES, "class_names": CLASS_NAMES,
        "class_counts": np.bincount(labels, minlength=7).tolist(),
        "test_class_counts": np.bincount(labels[test_ids], minlength=7).tolist(),
        "subset_class_counts": {str(size): np.bincount(labels[ids], minlength=7).tolist()
                                for size, ids in subsets.items()},
        "missing_values": 0, "scaling": "none; tree models use original numeric features",
        "raw_sha256": sha256(source), "sha256": {path.name: sha256(path) for path in files},
        "preparation_seconds": time.perf_counter() - begin,
    }
    write_json(manifest, description)
    print(f"数据已准备，共 {len(frame):,} 行，固定测试集 {len(test_ids):,} 行。")


if __name__ == "__main__":
    main()
