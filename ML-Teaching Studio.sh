#!/bin/sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LAUNCHER="$SCRIPT_DIR/scripts/launch_studio.py"

if [ -n "${ML_TEACHING_STUDIO_PYTHON:-}" ]; then
    PYTHON_BIN="$ML_TEACHING_STUDIO_PYTHON"
elif [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
else
    echo "No Python interpreter was found for ML-Teaching Studio." >&2
    exit 1
fi

cd "$SCRIPT_DIR" || exit 1
exec "$PYTHON_BIN" "$LAUNCHER" "$@"
