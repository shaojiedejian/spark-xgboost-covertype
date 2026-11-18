#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
exec "$PYSPARK_PYTHON" -m forest_project.predict "$@"
