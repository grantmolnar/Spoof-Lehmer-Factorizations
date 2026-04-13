#!/usr/bin/env python3
"""Run the full forest-property analysis suite against an existing
census database. Covers census, primitives, sporadic, and coverage
reports.

The database must already be populated. Use reproduce_census.py or
enumerate_all.py to create it.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

from spoof_lehmer.analysis import (  # noqa: E402
    analyze_primitives, analyze_sporadic_seeds, compute_coverage,
    format_coverage_report, format_primitives_report, format_report,
    format_sporadic_report, run_census,
)
from spoof_lehmer.storage import SQLiteRepository  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--db", type=Path, default=DATA_DIR / "census.db")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: database not found: {args.db}", file=sys.stderr)
        print("Run reproduce_census.py or enumerate_all.py first.", file=sys.stderr)
        sys.exit(1)

    bar = "=" * 62
    print(bar)
    print(f"Forest analysis for k={args.k} on {args.db}")
    print(bar)

    repo = SQLiteRepository(args.db)
    from collections.abc import Callable
    reports: list[tuple[str, Callable[[], str]]] = [
        ("census", lambda: format_report(run_census(repo, args.k))),
        ("primitives", lambda: format_primitives_report(analyze_primitives(repo, args.k))),
        ("sporadic", lambda: format_sporadic_report(analyze_sporadic_seeds(repo, args.k))),
        ("coverage", lambda: format_coverage_report(compute_coverage(repo, args.k))),
    ]
    for name, render in reports:
        print(f"\n----- {name} -----")
        print(render())
    repo.close()


if __name__ == "__main__":
    main()
