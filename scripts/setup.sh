#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
if [[ ! -x "$PYSPARK_PYTHON" ]]; then python3 -m venv .runtime/xgb-venv; fi
if command -v uv >/dev/null 2>&1; then
  UV_HTTP_TIMEOUT=120 uv pip install --python "$PYSPARK_PYTHON" -r requirements.txt
else
  "$PYSPARK_PYTHON" -m pip install --timeout 120 -r requirements.txt
fi
"$PYSPARK_PYTHON" -m pip freeze > requirements-lock.txt
"$PYSPARK_PYTHON" -c 'import pyspark, xgboost; print("Spark", pyspark.__version__, "XGBoost", xgboost.__version__)'
