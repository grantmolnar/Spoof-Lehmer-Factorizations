"""CLI for running analysis reports against an existing census database."""
from __future__ import annotations
import argparse
from pathlib import Path
from spoof_lehmer.analysis import (
    run_census,
    format_report,
    analyze_primitives,
    format_primitives_report,
    analyze_sporadic_seeds,
    format_sporadic_report,
    compute_coverage,
    format_coverage_report,
)
from spoof_lehmer.storage import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run analysis reports against an existing census database. "
            "Read-only: never mutates the database. Run `spoof-census` "
            "first to populate it."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Lehmer parameter k (default: 2). Must match a value used by spoof-census.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/census.db"),
        help="Path to the SQLite database (default: data/census.db).",
    )
    parser.add_argument(
        "--report",
        choices=["census", "primitives", "sporadic", "coverage", "all"],
        default="all",
        help=(
            "Which report to generate. "
            "'census': forest property check, distribution by length r. "
            "'primitives': structural invariants of non-derived Lehmer factorizations. "
            "'sporadic': Fermat / Fermat-derived / sporadic seed classification "
            "and the inclusion graph among plus-seeds. "
            "'coverage': exhaustiveness ledger - how far out the search has "
            "provably found everything, and what remains in the cascade pending queue. "
            "'all': all four (default)."
        ),
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: database {args.db} does not exist.")
        print("Run `spoof-census` first to populate it.")
        return

    repo = SQLiteRepository(args.db)

    if args.report in ("census", "all"):
        print(format_report(run_census(repo, args.k)))
        print()

    if args.report in ("coverage", "all"):
        print(format_coverage_report(compute_coverage(repo, args.k)))
        print()

    if args.report in ("primitives", "all"):
        print(format_primitives_report(analyze_primitives(repo, args.k)))
        print()

    if args.report in ("sporadic", "all"):
        print(format_sporadic_report(analyze_sporadic_seeds(repo, args.k)))
        print()

    repo.close()


if __name__ == "__main__":
    main()
