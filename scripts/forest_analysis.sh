#!/usr/bin/env bash
# Run the full forest-property analysis suite against an existing
# census database.
#
# Covers:
#   - census:    forest property check + distribution by r
#   - primitives: which factorizations are primitive vs derived via descent
#   - sporadic:  plus-seeds that aren't accounted for by known cascades
#   - coverage:  how many Lehmer factorizations are reached from each seed
#
# Usage:
#   scripts/forest_analysis.sh                          # k=2, data/census.db
#   DB=data/reproduce_k2_r6.db scripts/forest_analysis.sh
#   K=3 DB=data/reproduce_k3_r5.db scripts/forest_analysis.sh
#
# Env knobs:
#   K    Lehmer parameter (default 2)
#   DB   Input SQLite path (default data/census.db)
#
# The database must already be populated. Use scripts/reproduce_census.sh
# or `spoof-census` directly to create it.

set -euo pipefail

POETRY_RUN="${POETRY_RUN:-poetry run}"
K="${K:-2}"
DB="${DB:-data/census.db}"

if [[ ! -f "$DB" ]]; then
    echo "Error: database not found: $DB" >&2
    echo "Run scripts/reproduce_census.sh first, or set DB=... to an existing census." >&2
    exit 1
fi

echo "=============================================================="
echo "Forest analysis for k=${K} on ${DB}"
echo "=============================================================="

for report in census primitives sporadic coverage; do
    echo
    echo "----- ${report} -----"
    $POETRY_RUN spoof-analyze --k "$K" --db "$DB" --report "$report"
done
