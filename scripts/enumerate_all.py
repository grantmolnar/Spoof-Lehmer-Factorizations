#!/usr/bin/env python3
"""Enumerate every k-Lehmer factorization for a given length r and
parity, across the full k-range that could possibly produce solutions.

Streams output as factorizations are discovered, with r-transition
banners, per-level timing, and interrupt-safe resume via search_runs
bookkeeping. Reproduces the behavior of the original Molnar-Singh
spoof-Lehmer-factorizations repository.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

from spoof_lehmer.cli.enumerate import k_upper_bound  # noqa: E402
from spoof_lehmer.domain import Factorization  # noqa: E402
from spoof_lehmer.search import BoundsPropagationStrategy  # noqa: E402
from spoof_lehmer.storage import SQLiteRepository  # noqa: E402

STRATEGY_NAME = "bounds_propagation"


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:6.2f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m):>2d}m{s:05.2f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):d}h{int(m):02d}m{s:05.2f}s"


def plan_resume(
    repo: SQLiteRepository, max_r: int, is_even: bool,
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    dict[tuple[int, int], float],
]:
    """Classify every (r, k) pair we intend to search.

    Returns (completed, interrupted, prior_times):
      - completed: runs with status 'complete'. Skip these on resume.
      - interrupted: runs with status 'interrupted'. Re-run these;
        the previous attempt didn't finish.
      - prior_times: wall-clock seconds each completed (r, k) took,
        derived from the persisted started_at/finished_at timestamps.
        Used so the final summary reports times even when most pairs
        were resumed from cache and not re-run this session.
    A pair present in both completed and interrupted (rerun succeeded
    after crash) is treated as complete.
    """
    completed: set[tuple[int, int]] = set()
    interrupted: set[tuple[int, int]] = set()
    prior_times: dict[tuple[int, int], float] = {}
    for k in range(2, k_upper_bound(max_r, is_even) + 1):
        for run in repo.all_runs(k):
            if run.strategy != STRATEGY_NAME:
                continue
            pair = (int(run.max_r or 0), k)
            if run.status == "complete":
                completed.add(pair)
                prior_times[pair] = run.elapsed_seconds
            else:
                interrupted.add(pair)
    interrupted -= completed
    return completed, interrupted, prior_times


def dump_json(
    repo: SQLiteRepository, path: Path, max_r: int, is_even: bool,
) -> int:
    """Write every discovered factorization up to max_r as JSON.
    Atomic via tmp + rename.
    """
    payload: list[dict[str, object]] = []
    for r in range(2, max_r + 1):
        for k in range(2, k_upper_bound(r, is_even) + 1):
            for fact in repo.all_lehmers(k):
                if fact.length == r:
                    payload.append({
                        "k": k, "r": r, "factors": list(fact.factors),
                    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return len(payload)


def search_one_pair(
    repo: SQLiteRepository, r: int, k: int, is_even: bool,
    use_finishing_feasibility: bool,
    on_found: Callable[[int, Factorization], None],
) -> tuple[int, int]:
    """Search a single (r, k) pair with interrupt-safe bookkeeping.

    Writes an in-progress marker before discover() and abandons it
    after a clean return (the strategy wrote its own COMPLETE row).
    On Ctrl-C the marker persists for plan_resume to see.
    """
    run_id = repo.begin_run(STRATEGY_NAME, k, r)
    strategy = BoundsPropagationStrategy(
        k_target=k, max_r=r, min_r=r, is_even=is_even,
        use_finishing_feasibility=use_finishing_feasibility,
        on_found=lambda fact: on_found(k, fact),
    )
    result = strategy.discover(repo)
    repo.abandon_run(run_id)  # strategy already wrote COMPLETE row
    return result.added, result.nodes_explored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-r", type=int, default=6)
    parser.add_argument("--parity", choices=("odd", "even"), default="odd")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--timings", type=Path, default=None,
                        help="Optional TSV dump of per-(r, k) wall-clock "
                             "times in seconds, derived from search_runs.")
    parser.add_argument("--no-finishing-feasibility", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete the existing database before starting. "
                             "Default: resume.")
    args = parser.parse_args()

    is_even = args.parity == "even"
    db_path = args.db or DATA_DIR / f"enumerate_{args.parity}_r{args.max_r}.db"
    json_path = args.json or DATA_DIR / f"enumerate_{args.parity}_r{args.max_r}.json"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and db_path.exists():
        db_path.unlink()

    bar = "=" * 62
    print(bar)
    print(f"All-k enumeration: parity={args.parity}, r<={args.max_r}")
    print(bar)
    print(f"Database: {db_path} ({'fresh' if args.fresh else 'resume'})",
          flush=True)

    repo = SQLiteRepository(db_path)
    completed, interrupted, prior_times = plan_resume(repo, args.max_r, is_even)
    if interrupted:
        print()
        print("WARNING: prior interrupted runs detected; will re-search:")
        for r_i, k_i in sorted(interrupted):
            print(f"  (r={r_i}, k={k_i})")
        print()

    start_time = time.perf_counter()
    total_added = 0
    total_nodes = 0
    per_r_totals: dict[int, int] = {}
    per_r_times: dict[int, float] = {}
    # Per-(r, k) seconds for this session, fresh + resumed combined.
    pair_times: dict[tuple[int, int], float] = dict(prior_times)

    def on_found(k: int, fact: Factorization) -> None:
        elapsed = time.perf_counter() - start_time
        factors = ", ".join(str(x) for x in fact.factors)
        print(f"    [{_fmt_elapsed(elapsed)}] k={k:<4} found: ({factors})",
              flush=True)

    for r in range(2, args.max_r + 1):
        k_max_r = k_upper_bound(r, is_even)
        print()
        print(f"----- r = {r}   (k in [2, {k_max_r}]) -----", flush=True)
        r_added = 0
        r_nodes = 0
        r_resumed = 0
        r_start = time.perf_counter()
        for k in range(2, k_max_r + 1):
            if (r, k) in completed:
                r_resumed += 1
                continue
            pair_start = time.perf_counter()
            added, nodes = search_one_pair(
                repo, r, k, is_even,
                use_finishing_feasibility=not args.no_finishing_feasibility,
                on_found=on_found,
            )
            pair_times[(r, k)] = time.perf_counter() - pair_start
            r_added += added
            r_nodes += nodes
        r_elapsed = time.perf_counter() - r_start
        per_r_totals[r] = r_added
        per_r_times[r] = r_elapsed
        total_elapsed = time.perf_counter() - start_time
        skipped_label = f" ({r_resumed} k resumed)" if r_resumed else ""
        print(
            f"  r = {r} complete: {r_added} new factorizations, "
            f"{r_nodes:,} nodes, took {_fmt_elapsed(r_elapsed)}  "
            f"(total {_fmt_elapsed(total_elapsed)}){skipped_label}",
            flush=True,
        )
        total_added += r_added
        total_nodes += r_nodes
        dump_json(repo, json_path, args.max_r, is_even)
        print(f"  JSON checkpointed to {json_path}", flush=True)

    grand_total = time.perf_counter() - start_time
    print()
    print(bar)
    print(f"Totals: {total_added} new factorizations across "
          f"{total_nodes:,} nodes in {_fmt_elapsed(grand_total)}.")
    for r, n in sorted(per_r_totals.items()):
        t = per_r_times[r]
        print(f"  r = {r}: {n:>4} factorizations   ({_fmt_elapsed(t)})")
    count = dump_json(repo, json_path, args.max_r, is_even)
    print(f"Wrote JSON ({count} factorizations total) to {json_path}")

    if args.timings is not None:
        args.timings.parent.mkdir(parents=True, exist_ok=True)
        lines = ["r\tk\tseconds"]
        for (r, k), secs in sorted(pair_times.items()):
            lines.append(f"{r}\t{k}\t{secs:.3f}")
        args.timings.write_text("\n".join(lines) + "\n")
        print(f"Wrote timings ({len(pair_times)} pairs) to {args.timings}")

    repo.close()


if __name__ == "__main__":
    main()
