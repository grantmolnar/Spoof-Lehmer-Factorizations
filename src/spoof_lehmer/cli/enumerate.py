"""CLI that enumerates every k-Lehmer factorization for a given
length r and parity, across the full k-range that could possibly
produce solutions.

This reproduces the behavior of the original Molnar-Singh repository
(https://github.com/grantmolnar/Spoof-Lehmer-Factorizations), which
enumerated all spoofs for a given (r, parity) and saved them.

Math: a k-Lehmer factorization with r factors whose smallest factor is
at least a_min satisfies k < (a_min / (a_min - 1))^r, because
k = (eps - 1)/phi* < prod(x / (x-1)) <= (a_min/(a_min-1))^r.
So the relevant k-range is finite:
    odd   (a_min = 3): k in {2, ..., floor((3/2)^r)}
    even  (a_min = 2): k in {2, ..., 2^r - 1}
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from spoof_lehmer.search import (
    BoundsPropagationStrategy,
    StderrProgressReporter,
)
from spoof_lehmer.storage import SQLiteRepository


def k_upper_bound(r: int, is_even: bool) -> int:
    """Largest k worth checking for r-factor spoofs of given parity.

    Derivation: k < (a_min / (a_min - 1))^r. Take floor and ensure >= 2.
    """
    if is_even:
        # (2/1)^r = 2^r; k < 2^r means k <= 2^r - 1.
        return int(max(2, 2**r - 1))
    # (3/2)^r; use integer arithmetic: floor(3^r / 2^r).
    return int(max(2, 3**r // 2**r))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate every k-Lehmer factorization of a given length r "
            "and parity, across all k values that could possibly produce "
            "solutions. Reproduces the behavior of the original "
            "Molnar-Singh spoof-Lehmer-factorizations repository."
        ),
    )
    parser.add_argument(
        "--max-r",
        type=int,
        required=True,
        help="Enumerate at every length r' in [2, max-r].",
    )
    parser.add_argument(
        "--parity",
        choices=("odd", "even"),
        default="odd",
        help=(
            "'odd' (default): factors must be odd and >= 3 (the classical "
            "Lehmer setting). 'even': factors must be >= 2, any parity."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "SQLite database to append results to. Default is "
            "data/enumerate_{parity}_r{max_r}.db."
        ),
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help=(
            "If set, also write a JSON summary listing every discovered "
            "factorization as {k, r, factors}."
        ),
    )
    parser.add_argument(
        "--no-finishing-feasibility",
        action="store_true",
        help="Disable the closed-form endgame (debug/cross-check only).",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help=(
            "Emit human-readable progress to stderr: per-k timing, "
            "per-length timing, each factorization as it's found, "
            "and a throttled heartbeat during long searches. "
            "Default: on when stderr is a TTY, off otherwise."
        ),
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Suppress the stderr progress stream.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help=(
            "Minimum wall-clock seconds between heartbeat lines during "
            "a single (k, r) search. Set to 0 to disable heartbeats "
            "while keeping per-length and per-found output. Default 30."
        ),
    )
    args = parser.parse_args()

    is_even = args.parity == "even"
    db_path = args.db or Path(
        f"data/enumerate_{args.parity}_r{args.max_r}.db"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Progress-reporter selection:
    #   --progress on:  StderrProgressReporter with given heartbeat.
    #   --no-progress:  None (strategy will use SilentProgressReporter).
    #   default (None): auto-detect TTY on stderr.
    if args.progress is None:
        enable_progress = sys.stderr.isatty()
    else:
        enable_progress = args.progress
    progress = (
        StderrProgressReporter(heartbeat_seconds=args.heartbeat_seconds)
        if enable_progress
        else None
    )

    k_max = k_upper_bound(args.max_r, is_even)
    print(f"Parity: {args.parity}  |  max r: {args.max_r}  |  k range: [2, {k_max}]")
    print(f"Database: {db_path}")
    if enable_progress:
        print(f"Progress: on (heartbeat every {args.heartbeat_seconds:g}s on stderr)")
    print()

    repo = SQLiteRepository(db_path)
    total_added = 0
    total_nodes = 0
    by_k: dict[int, int] = {}

    for k in range(2, k_max + 1):
        strat = BoundsPropagationStrategy(
            k_target=k,
            max_r=args.max_r,
            is_even=is_even,
            use_finishing_feasibility=not args.no_finishing_feasibility,
            progress=progress,
        )
        result = strat.discover(repo)
        by_k[k] = result.added
        total_added += result.added
        total_nodes += result.nodes_explored
        marker = "" if result.added == 0 else f"  (+{result.added})"
        print(
            f"  k = {k:>4}: added={result.added:>3}  "
            f"nodes={result.nodes_explored:>10,}{marker}"
        )

    print()
    print(f"Total: {total_added} new factorizations across {total_nodes:,} nodes.")
    nontrivial = {k: v for k, v in by_k.items() if v > 0}
    if nontrivial:
        summary = ", ".join(f"k={k}:{n}" for k, n in sorted(nontrivial.items()))
        print(f"Nontrivial k values: {summary}")

    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for k in range(2, k_max + 1):
            for fact in repo.all_lehmers(k):
                payload.append({
                    "k": k,
                    "r": fact.length,
                    "factors": list(fact.factors),
                })
        args.dump_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON summary ({len(payload)} factorizations) to {args.dump_json}")

    repo.close()


if __name__ == "__main__":
    main()
