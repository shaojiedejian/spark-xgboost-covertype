#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
"$PYSPARK_PYTHON" scripts/prepare_data.py
"$PYSPARK_PYTHON" tests/verify_data.py
"$PYSPARK_PYTHON" scripts/run_experiments.py
"$PYSPARK_PYTHON" tests/verify_results.py
"$PYSPARK_PYTHON" scripts/make_report.py
