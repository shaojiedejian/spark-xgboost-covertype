"""项目配置、校验值和结果写入。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "elevation", "aspect", "slope", "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology", "horizontal_distance_to_roadways",
    "hillshade_9am", "hillshade_noon", "hillshade_3pm", "horizontal_distance_to_fire_points",
] + [f"wilderness_{i}" for i in range(1, 5)] + [f"soil_{i}" for i in range(1, 41)]
CLASS_NAMES = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
               "Aspen", "Douglas-fir", "Krummholz"]


def config():
    return json.loads((ROOT / "configs/experiment.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
