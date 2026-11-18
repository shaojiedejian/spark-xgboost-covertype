#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
if [[ -z "${FOREST_JAVA_HOME:-}" && "$(uname)" == "Darwin" ]]; then
  FOREST_JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null || true)"
fi
if [[ -n "${FOREST_JAVA_HOME:-}" ]]; then
  export JAVA_HOME="$FOREST_JAVA_HOME"
  export PATH="$JAVA_HOME/bin:$PATH"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYSPARK_PYTHON="$PROJECT_ROOT/.runtime/xgb-venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
export SPARK_LOCAL_IP=127.0.0.1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PROJECT_ROOT/.runtime/matplotlib"
export SPARK_LOCAL_DIRS="$PROJECT_ROOT/.runtime/spark-tmp"
mkdir -p "$SPARK_LOCAL_DIRS" "$MPLCONFIGDIR"
