#!/usr/bin/env python3
"""Analyze family structure of the r<=7 census.

Groups r=7 k=2 factorizations by length-5 prefix, identifying:
  - The Fermat-5 family (prefix (3,5,17,257,65537), a plus-seed)
  - Sporadic plus-seed families (other prefixes with > 1 completion)
  - Unique-prefix factorizations (isolated leaves)

Reads data/enumerate_odd_r7.json and prints a structured report.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def is_plus_seed(prefix: tuple[int, ...], k: int) -> bool:
    """A k-plus-seed satisfies kP = E + 1, equivalently A = kP - E = 1.

    Plus-seeds are the structural origin of multi-completion families:
    when A = 1, the s = r-2 finishing-feasibility identity
    (Aa - B)(Ab - B) = BE - A specializes to (a - B)(b - B) = BE - 1,
    which has many divisor-pair solutions when BE - 1 is composite.
    For non-plus-seeds (A >= 2), the divisibility-mod-A constraint
    on (a, b) typically eliminates all but 0 or 1 divisor pairs.
    """
    eps = 1
    phi = 1
    for x in prefix:
        eps *= x
        phi *= (x - 1)
    return k * phi == eps + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "enumerate_odd_r7.json",
        help="Path to enumerate_odd_r7.json",
    )
    args = ap.parse_args()

    data = json.loads(args.data.read_text())

    by_rk: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    for entry in data:
        by_rk[(entry["r"], entry["k"])].append(tuple(entry["factors"]))

    print(f"Total factorizations: {len(data)}")
    print()
    print("Distribution by (r, k):")
    print(f"  {'r':>2}  {'k':>2}  count")
    for (r, k), facts in sorted(by_rk.items()):
        print(f"  {r:>2}  {k:>2}  {len(facts):>4}")
    print()

    # Family analysis: r=7 k=2
    r7k2 = by_rk.get((7, 2), [])
    print(f"Family analysis: r=7, k=2 ({len(r7k2)} factorizations)")
    print()

    by_prefix: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
    for f in r7k2:
        by_prefix[f[:5]].append(f)

    multi = sorted(
        ((p, fs) for p, fs in by_prefix.items() if len(fs) > 1),
        key=lambda x: (-len(x[1]), x[0]),
    )
    singletons = [(p, fs[0]) for p, fs in by_prefix.items() if len(fs) == 1]

    print(f"  Length-5 prefixes producing multiple completions ({len(multi)} prefixes):")
    print()
    for prefix, completions in multi:
        is_seed = is_plus_seed(prefix, 2)
        seed_marker = " [plus-seed]" if is_seed else ""
        print(f"    {prefix}: {len(completions)} completions{seed_marker}")
        x6_values = sorted({c[5] for c in completions})
        print(f"      x_6 range: [{min(x6_values)}, {max(x6_values)}]")
        if len(completions) <= 4:
            for c in completions:
                print(f"        {c}")
    print()

    print(f"  Length-5 prefixes with unique completion: {len(singletons)}")
    print()

    # Other-k r=7
    r7_other = [(k, by_rk.get((7, k), [])) for k in [3, 4, 5, 6, 7, 8, 9, 10]]
    r7_other = [(k, fs) for k, fs in r7_other if fs]
    if r7_other:
        print("r=7 at k != 2:")
        for k, fs in r7_other:
            for f in fs:
                print(f"  k={k}: {f}")


if __name__ == "__main__":
    main()
