#!/usr/bin/env bash
# Reproduce the Molnar-Singh r <= 6 census of k-Lehmer factorizations.
#
# Runs the bounds-propagation strategy (with finishing-feasibility) from
# a fresh SQLite database, then cross-checks against the recurrence
# strategy, then prints the census / forest / primitives / sporadic /
# coverage reports.
#
# Usage:
#   scripts/reproduce_census.sh          # k=2, r<=6 (default)
#   K=3 MAX_R=5 scripts/reproduce_census.sh
#   DB=data/my_run.db scripts/reproduce_census.sh
#
# Env knobs:
#   K       Lehmer parameter (default 2)
#   MAX_R   Maximum factorization length (default 6)
#   DB      Output SQLite path (default data/reproduce_k${K}_r${MAX_R}.db)
#   MAX_N   --max-N for the recurrence cross-check (default 1e13)
#
# Assumes the project is installed under Poetry: run `poetry install` once,
# then invoke this script (it uses `poetry run` so it works without
# activating the shell via `poetry shell`).

set -euo pipefail

POETRY_RUN="${POETRY_RUN:-poetry run}"
K="${K:-2}"
MAX_R="${MAX_R:-6}"
MAX_N="${MAX_N:-1e13}"
DB="${DB:-data/reproduce_k${K}_r${MAX_R}.db}"

mkdir -p "$(dirname "$DB")"
if [[ -f "$DB" ]]; then
    echo "Removing stale database: $DB"
    rm -f "$DB"
fi

echo "=============================================================="
echo "Reproducing k=${K} r<=${MAX_R} census at ${DB}"
echo "=============================================================="

echo
echo "[1/3] Bounds propagation (finishing-feasibility enabled)"
$POETRY_RUN spoof-census \
    --strategy bounds \
    --k "$K" \
    --max-r "$MAX_R" \
    --db "$DB" \
    --no-cascade

echo
echo "[2/3] Cross-check: snapshot bounds set, then run recurrence into same DB"
$POETRY_RUN python - "$DB" "$K" <<'PY' >/tmp/bounds_snapshot.txt
import sqlite3, sys
path, k = sys.argv[1], int(sys.argv[2])
conn = sqlite3.connect(path)
rows = conn.execute(
    "SELECT factors FROM factorizations WHERE k=? AND kind='lehmer'",
    (k,),
).fetchall()
conn.close()
for (f,) in sorted(rows):
    print(f)
PY

$POETRY_RUN spoof-census \
    --strategy recurrence \
    --k "$K" \
    --max-r "$MAX_R" \
    --max-N "$MAX_N" \
    --db "$DB" \
    --no-cascade

$POETRY_RUN python - "$DB" "$K" /tmp/bounds_snapshot.txt <<'PY'
import sqlite3, sys
path, k, snap_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(snap_path) as fh:
    bounds = {line.rstrip("\n") for line in fh if line.strip()}
conn = sqlite3.connect(path)
rows = conn.execute(
    "SELECT factors FROM factorizations WHERE k=? AND kind='lehmer'",
    (k,),
).fetchall()
conn.close()
final = {r[0] for r in rows}
only_recurrence = final - bounds
print(f"  bounds:     {len(bounds)} lehmers")
print(f"  union:      {len(final)} lehmers after recurrence")
if only_recurrence:
    print(f"  MISMATCH: recurrence found {len(only_recurrence)} factorizations bounds missed:")
    for s in sorted(only_recurrence)[:5]:
        print(f"    {s}")
    sys.exit(1)
print("  AGREE: recurrence added nothing new; bounds was exhaustive.")
PY
rm -f /tmp/bounds_snapshot.txt

echo
echo "[3/3] Census + forest report (authoritative)"
$POETRY_RUN spoof-analyze --k "$K" --db "$DB" --report census

echo
echo "Done. Database: $DB"
echo "Run scripts/forest_analysis.sh with DB=$DB for deeper forest analysis."
