#!/usr/bin/env python3
"""Reproduce the Molnar-Singh k-Lehmer census from scratch at a fixed k.

Three phases:
  1. BoundsPropagationStrategy (finishing-feasibility enabled) into a
     fresh SQLite database.
  2. Cross-check: run RecurrenceStrategy into the same database and
     assert it added nothing new (bounds was exhaustive).
  3. Print the census + forest-property report.

Unlike enumerate_all.py, this targets a SINGLE k at a time. For the
all-k analog of Grant's original repo behavior, use enumerate_all.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

from spoof_lehmer.analysis import format_report, run_census  # noqa: E402
from spoof_lehmer.search import (  # noqa: E402
    BoundsPropagationStrategy, RecurrenceStrategy,
)
from spoof_lehmer.storage import SQLiteRepository  # noqa: E402


def lehmer_set(repo: SQLiteRepository, k: int) -> set[tuple[int, ...]]:
    return {f.factors for f in repo.all_lehmers(k)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-r", type=int, default=6)
    parser.add_argument("--max-N", type=float, default=1e13,
                        help="--max-N for the recurrence cross-check (default 1e13).")
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    db_path = args.db or DATA_DIR / f"reproduce_k{args.k}_r{args.max_r}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    bar = "=" * 62
    print(bar)
    print(f"Reproducing k={args.k} r<={args.max_r} census at {db_path}")
    print(bar)

    repo = SQLiteRepository(db_path)

    print("\n[1/3] Bounds propagation (finishing-feasibility enabled)")
    bounds_result = BoundsPropagationStrategy(
        k_target=args.k, max_r=args.max_r,
    ).discover(repo)
    print(f"  added {bounds_result.added} factorizations "
          f"({bounds_result.nodes_explored:,} nodes)")
    bounds_set = lehmer_set(repo, args.k)

    print("\n[2/3] Recurrence cross-check into same DB (populates coverage ledger)")
    rec_result = RecurrenceStrategy(
        k=args.k, max_r=args.max_r, max_N=int(args.max_N),
    ).discover(repo)
    print(f"  recurrence: added {rec_result.added}")
    union_set = lehmer_set(repo, args.k)
    only_rec = union_set - bounds_set
    print(f"  bounds:  {len(bounds_set)} lehmers")
    print(f"  union:   {len(union_set)} lehmers after recurrence")
    if only_rec:
        print(f"  MISMATCH: recurrence found {len(only_rec)} bounds missed:")
        for s in sorted(only_rec)[:5]:
            print(f"    {s}")
        sys.exit(1)
    print("  AGREE: recurrence added nothing new; bounds was exhaustive.")

    print("\n[3/3] Census + forest report")
    print(format_report(run_census(repo, args.k)))

    print(f"\nDone. Database: {db_path}")
    print(f"Run scripts/forest_analysis.py with --db {db_path} for deeper analysis.")
    repo.close()


if __name__ == "__main__":
    main()
