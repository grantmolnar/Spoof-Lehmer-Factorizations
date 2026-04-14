#!/usr/bin/env python3
"""Parallel enumeration of a single (r, k) search by length-2 prefix
partition.

Where `enumerate_all.py --jobs N` parallelizes ACROSS k (one worker
per (r, k) pair, useful when many ks are slow), this script
parallelizes WITHIN a single (r, k) by dispatching each length-2
prefix subtree to a separate worker. This is the right granularity
when k=2 dominates total runtime, which is the case at r=7 and r=8.

Usage:
  poetry run python scripts/enumerate_within_k.py \\
    --r 8 --k 2 --parity odd --jobs 8 [--prefix-length 2]

Streams finds as workers report them, prints per-prefix elapsed time,
and writes results to stdout (no SQLite repository involvement; this
is a research/exploration script, not the canonical census builder).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spoof_lehmer.search import BoundsPropagationStrategy  # noqa: E402
from spoof_lehmer.storage import InMemoryRepository  # noqa: E402


def enumerate_valid_prefixes(
    k: int, r: int, is_even: bool, prefix_length: int,
) -> list[tuple[int, ...]]:
    """Generate every length-`prefix_length` prefix that survives the
    bounds-propagation upper-bound test for a length-r k-Lehmer.

    Equivalent to: prefixes for which U(prefix) >= k. We compute this
    by trial-extending shorter prefixes; this is a single-process
    scan that's fast even for r=8 (the cap on x_1 is small).
    """
    min_first = 2 if is_even else 3
    step = 1 if is_even else 2

    def survives_upper_bound(prefix: tuple[int, ...]) -> bool:
        """Check that some completion of prefix to length r might
        satisfy k * phi*(F) = eps(F) - 1.
        """
        eps = 1
        phi = 1
        for x in prefix:
            eps *= x
            phi *= (x - 1)
        remaining = r - len(prefix)
        if remaining <= 0:
            return True
        # U-bound: smallest legal next factor is prefix[-1] (or
        # min_first if prefix is empty). Use that as 'a'.
        a = prefix[-1] if prefix else min_first
        # eps * a^remaining - 1 >= k * phi * (a-1)^remaining ?
        return bool(eps * a**remaining - 1 >= k * phi * (a - 1) ** remaining)

    def extend(prefix: tuple[int, ...]) -> list[tuple[int, ...]]:
        if len(prefix) == prefix_length:
            return [prefix]
        out: list[tuple[int, ...]] = []
        start = prefix[-1] if prefix else min_first
        # Cap the candidate range: we walk x upward and stop as soon as
        # adding x to the prefix fails the U-bound. This mirrors the
        # main strategy's termination logic.
        x = start
        while True:
            extended = (*prefix, x)
            if not survives_upper_bound(extended):
                break
            out.extend(extend(extended))
            x += step
            # Defensive cap to avoid pathological infinite loops in
            # degenerate parameter regions.
            if x > 10**6:
                break
        return out

    return extend(())


def _worker_search(
    r: int, k: int, is_even: bool, prefix: tuple[int, ...],
) -> tuple[tuple[int, ...], list[tuple[int, ...]], float, int]:
    """Run a subtree search and return primitive results."""
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=k, max_r=r, min_r=r, is_even=is_even,
    )
    t0 = time.perf_counter()
    result = strat.discover_subtree(repo, prefix)
    elapsed = time.perf_counter() - t0
    factor_tuples = [fact.factors for fact in repo.all_lehmers(k)]
    return prefix, factor_tuples, elapsed, result.nodes_explored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--parity", choices=("odd", "even"), default="odd")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--prefix-length", type=int, default=2,
        help="Length of prefix to partition on (default 2). Larger "
             "gives finer-grained tasks but more overhead.",
    )
    args = parser.parse_args()

    is_even = args.parity == "even"

    print(f"Generating valid length-{args.prefix_length} prefixes "
          f"for r={args.r}, k={args.k}, parity={args.parity}...",
          flush=True)
    prefixes = enumerate_valid_prefixes(
        args.k, args.r, is_even, args.prefix_length,
    )
    print(f"  {len(prefixes)} prefixes will be searched in parallel.",
          flush=True)

    if not prefixes:
        print("No valid prefixes; nothing to do.")
        return

    start = time.perf_counter()
    total_added = 0
    total_nodes = 0
    all_factorizations: set[tuple[int, ...]] = set()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.jobs, len(prefixes)), mp_context=ctx,
    ) as pool:
        futures = {
            pool.submit(
                _worker_search, args.r, args.k, is_even, prefix,
            ): prefix
            for prefix in prefixes
        }
        for fut in as_completed(futures):
            prefix, factor_tuples, elapsed, nodes = fut.result()
            wall = time.perf_counter() - start
            new = 0
            for ft in factor_tuples:
                if ft not in all_factorizations:
                    all_factorizations.add(ft)
                    new += 1
                    print(
                        f"    [{wall:7.2f}s] found: {ft}",
                        flush=True,
                    )
            total_added += new
            total_nodes += nodes
            print(
                f"  prefix {prefix} done in {elapsed:.2f}s, "
                f"{new} found, {nodes:,} nodes",
                flush=True,
            )

    grand = time.perf_counter() - start
    print()
    print(f"Total: {total_added} factorizations across "
          f"{total_nodes:,} nodes in {grand:.2f}s.")


if __name__ == "__main__":
    main()
