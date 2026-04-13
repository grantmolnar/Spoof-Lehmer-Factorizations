"""Top-level CLI commands."""
from __future__ import annotations
import argparse
from pathlib import Path
from spoof_lehmer.analysis import run_census, format_report
from spoof_lehmer.factoring import default_backend
from spoof_lehmer.search import (
    BoundsPropagationStrategy,
    CascadeStrategy,
    RecurrenceStrategy,
)
from spoof_lehmer.storage import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a census of k-Lehmer factorizations and plus-seeds. "
            "A k-Lehmer factorization is a multiset F = {x_1, ..., x_r} "
            "of integers >= 2 satisfying k * prod(x_i - 1) = prod(x_i) - 1. "
            "Results accumulate in a SQLite database across runs."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help=(
            "Lehmer parameter k in the equation k*phi*(F) = eps(F) - 1. "
            "k=2 is the classical case (default)."
        ),
    )
    parser.add_argument(
        "--max-r",
        type=int,
        default=7,
        help=(
            "Maximum factorization length r = |F| for the bottom-up search. "
            "Search cost grows roughly exponentially in r (default: 7)."
        ),
    )
    parser.add_argument(
        "--max-N",
        type=float,
        default=1e13,
        help=(
            "Maximum evaluation N = eps(F) = prod(x_i) for the bottom-up "
            "search. Factorizations with product exceeding this are pruned "
            "(default: 1e13)."
        ),
    )
    parser.add_argument(
        "--node-limit",
        type=int,
        default=50_000_000,
        help="Maximum search-tree nodes per (length, target) pair (default: 50M).",
    )
    parser.add_argument(
        "--cascade-rounds",
        type=int,
        default=0,
        help=(
            "Rounds of top-down extension. 0 (default) means run until "
            "natural termination - the loop stops when a round produces no "
            "new seeds. A positive integer caps the number of rounds for "
            "predictable wall time. Each round factors Delta_L(N) = N^2+N-1 "
            "(Lehmer extensions) and Delta_S(N) = N^2+N+1 (seed-to-seed "
            "extensions) for every known plus-seed."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/census.db"),
        help="Path to the SQLite database (default: data/census.db).",
    )
    parser.add_argument(
        "--strategy",
        choices=("recurrence", "bounds"),
        default="recurrence",
        help=(
            "Bottom-up search strategy. 'recurrence' (default) is the "
            "original bounded-N recurrence. 'bounds' is the "
            "bounds-propagation strategy with finishing-feasibility "
            "closed forms at s=r-1 and s=r-2; it ignores --max-N "
            "(termination is bounds-driven)."
        ),
    )
    parser.add_argument(
        "--no-finishing-feasibility",
        action="store_true",
        help=(
            "For --strategy bounds: disable the closed-form endgame at "
            "s=r-1 and s=r-2 and use plain bounds propagation throughout. "
            "Useful for cross-checking; normal runs leave this off."
        ),
    )
    parser.add_argument(
        "--no-recurrence",
        action="store_true",
        help="Skip the bottom-up recurrence search.",
    )
    parser.add_argument(
        "--no-cascade",
        action="store_true",
        help="Skip the top-down cascade search.",
    )
    args = parser.parse_args()

    repo = SQLiteRepository(args.db)
    backend = default_backend()
    print(f"Database: {args.db}")
    print(f"Factoring backend: {backend.name}")
    print()

    if not args.no_recurrence:
        if args.strategy == "bounds":
            strat = BoundsPropagationStrategy(
                k_target=args.k,
                max_r=args.max_r,
                use_finishing_feasibility=not args.no_finishing_feasibility,
            )
            ff_label = "" if args.no_finishing_feasibility else ", finishing-feasibility on"
            print(f"[1/2] Bounds propagation (max_r={args.max_r}{ff_label})")
            result = strat.discover(repo)
            status_tag = "" if result.is_complete else f"  [{result.status.value}]"
            print(f"      added {result.added} new factorizations"
                  f" ({result.nodes_explored:,} nodes){status_tag}")
        else:
            rec = RecurrenceStrategy(
                k=args.k,
                max_r=args.max_r,
                max_N=int(args.max_N),
                node_limit=args.node_limit,
            )
            print(f"[1/2] Bottom-up recurrence (max_r={args.max_r}, max_N={int(args.max_N):.0e})")
            result = rec.discover(repo)
            status_tag = "" if result.is_complete else f"  [{result.status.value}]"
            print(f"      added {result.added} new factorizations"
                  f" ({result.nodes_explored:,} nodes){status_tag}")

    if not args.no_cascade:
        rounds: int | None = args.cascade_rounds if args.cascade_rounds > 0 else None
        cas = CascadeStrategy(
            k=args.k,
            backend=backend,
            max_rounds=rounds,
        )
        rounds_label = f"{rounds} rounds" if rounds is not None else "until fixpoint"
        print(f"[2/2] Top-down cascade ({rounds_label})")
        result = cas.discover(repo)
        status_tag = "" if result.is_complete else f"  [{result.status.value}]"
        print(f"      added {result.added} new factorizations{status_tag}")

    print()
    report = run_census(repo, args.k)
    print(format_report(report))
    repo.close()


if __name__ == "__main__":
    main()
