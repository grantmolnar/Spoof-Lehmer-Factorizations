"""CLI for the top-down cascade extension step.

Runs the CascadeStrategy against an existing census database: for every
known plus-seed, factor Delta_L(N) = N^2 + N - 1 (to find new Lehmer
factorizations that descend from it) and Delta_S(N) = N^2 + N + 1 (to
find new plus-seeds that it seeds). Runs until the process reaches a
fixpoint, or for a capped number of rounds.

Read-optional: writes newly discovered factorizations back into the
database but never mutates existing rows. Run `spoof-census` first to
populate the initial seed set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from spoof_lehmer.factoring import default_backend
from spoof_lehmer.search import CascadeStrategy
from spoof_lehmer.storage import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Top-down cascade extension of known plus-seeds. Finds new "
            "Lehmer factorizations and new plus-seeds by factoring "
            "Delta_L(N) = N^2+N-1 and Delta_S(N) = N^2+N+1 for each seed "
            "already in the database."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Lehmer parameter k (default: 2). Must match a value used by spoof-census.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help=(
            "Number of cascade rounds. 0 (default) means run until natural "
            "termination - the loop stops when a round produces no new seeds. "
            "A positive integer caps the number of rounds for predictable "
            "wall time."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/census.db"),
        help="Path to the SQLite database (default: data/census.db).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: database {args.db} does not exist.")
        print("Run `spoof-census` first to populate it.")
        return

    repo = SQLiteRepository(args.db)
    backend = default_backend()
    print(f"Database: {args.db}")
    print(f"Factoring backend: {backend.name}")
    print()

    rounds: int | None = args.rounds if args.rounds > 0 else None
    strat = CascadeStrategy(k=args.k, backend=backend, max_rounds=rounds)
    rounds_label = f"{rounds} rounds" if rounds is not None else "until fixpoint"
    print(f"Cascade extension ({rounds_label}) for k={args.k}")
    result = strat.discover(repo)
    status_tag = "" if result.is_complete else f"  [{result.status.value}]"
    print(f"  added {result.added} new factorizations{status_tag}")
    repo.close()


if __name__ == "__main__":
    main()
