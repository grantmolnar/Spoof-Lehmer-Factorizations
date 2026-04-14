"""CLI to emit organized census tables from the JSON census.

Reads ``data/enumerate_odd_r7.json`` and emits LaTeX:
  * Table A (chain inventory) to stdout, then
  * Table B (full hierarchical longtable) following.

All organization and rendering logic lives in
:mod:`spoof_lehmer.analysis.census_organization` and
:mod:`spoof_lehmer.reporting.latex_census`; this script is purely the
glue that loads JSON and prints the result.

Usage::

    poetry run python scripts/emit_organized_census.py
    poetry run python scripts/emit_organized_census.py > paper/tables.tex
"""
from __future__ import annotations

import json
from pathlib import Path

from spoof_lehmer.analysis.census_organization import organize
from spoof_lehmer.reporting.latex_census import (
    render_chain_inventory_table,
    render_full_census,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    census_path = repo_root / "data" / "enumerate_odd_r7.json"
    raw = json.loads(census_path.read_text())

    # Filter to k = 2 entries.
    factorizations = [tuple(d["factors"]) for d in raw if d["k"] == 2]

    org = organize(factorizations, k=2)

    print("% --- Table A: Chain inventory ---")
    print(render_chain_inventory_table(org))
    print()
    print("% --- Table B: Full hierarchical census (longtable) ---")
    print("% Requires \\usepackage{longtable} in the preamble.")
    print(render_full_census(org))


if __name__ == "__main__":
    main()
