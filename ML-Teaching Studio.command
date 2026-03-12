#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/ML-Teaching Studio.sh" "$@"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo
    read -r -p "ML-Teaching Studio exited with status $STATUS. Press Return to close..." _
fi
exit "$STATUS"
