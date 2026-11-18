"""在首次训练之前检查 Java、Python 包及 XGBoost CPU 原生库。"""
import importlib
import json
import platform
import subprocess
import sys

packages = ["pyspark", "xgboost", "numpy", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib", "setuptools"]
versions = {}
for package in packages:
    module = importlib.import_module(package)
    versions[package] = getattr(module, "__version__", "available")
java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
print(json.dumps({"python": sys.version, "platform": platform.platform(), "java": java.stderr,
                  "packages": versions}, indent=2))
