#!/usr/bin/env bash
# Enumerate every k-Lehmer factorization for a given length r and
# parity, across the full k-range.
#
# Reproduces the behavior of the original Molnar-Singh spoof-Lehmer
# repository: exhaustive, saved to disk, for use in paper census
# tables.
#
# Usage:
#   scripts/enumerate_all.sh                   # odd, r<=6 (default)
#   MAX_R=7 scripts/enumerate_all.sh
#   PARITY=even MAX_R=5 scripts/enumerate_all.sh
#
# Env knobs:
#   MAX_R    Maximum factorization length (default 6)
#   PARITY   odd (default) or even
#   DB       Output SQLite (default data/enumerate_${PARITY}_r${MAX_R}.db)
#   JSON     Output JSON (default data/enumerate_${PARITY}_r${MAX_R}.json)

set -euo pipefail

POETRY_RUN="${POETRY_RUN:-poetry run}"
MAX_R="${MAX_R:-6}"
PARITY="${PARITY:-odd}"
DB="${DB:-data/enumerate_${PARITY}_r${MAX_R}.db}"
JSON="${JSON:-data/enumerate_${PARITY}_r${MAX_R}.json}"

mkdir -p "$(dirname "$DB")" "$(dirname "$JSON")"
rm -f "$DB"

echo "=============================================================="
echo "All-k enumeration: parity=${PARITY}, r<=${MAX_R}"
echo "=============================================================="
$POETRY_RUN spoof-enumerate \
    --max-r "$MAX_R" \
    --parity "$PARITY" \
    --db "$DB" \
    --dump-json "$JSON"
