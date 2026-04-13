"""Hasanalizade extension equations - both k-independent."""
from __future__ import annotations
from typing import Callable
from spoof_lehmer.domain.factorization import Factorization

# Type for a factoring backend: n -> {prime: exponent}
FactorFn = Callable[[int], dict[int, int]]


def lehmer_delta(N: int) -> int:
    """Delta_L(N) = N^2 + N - 1. Divisor pairs give Lehmer extensions."""
    return N * N + N - 1


def seed_delta(N: int) -> int:
    """Delta_S(N) = N^2 + N + 1. Divisor pairs give plus-seed extensions."""
    return N * N + N + 1


def divisor_pairs_from_factorization(factors: dict[int, int], n: int) -> list[tuple[int, int]]:
    """Enumerate all (d1, d2) with d1 <= d2, d1 * d2 = n."""
    divs = [1]
    for p, e in factors.items():
        divs = [d * p**i for d in divs for i in range(e + 1)]
    divs.sort()
    pairs = []
    lo, hi = 0, len(divs) - 1
    while lo <= hi:
        prod = divs[lo] * divs[hi]
        if prod == n:
            pairs.append((divs[lo], divs[hi]))
            lo += 1
            hi -= 1
        elif prod < n:
            lo += 1
        else:
            hi -= 1
    return pairs


def extensions_from_seed(
    seed: Factorization,
    factor_fn: FactorFn,
    target: str = "lehmer",
) -> list[Factorization]:
    """Generate all extensions of a plus-seed by factoring Delta(N).

    Args:
        seed: a plus-seed.
        factor_fn: backend that factors integers.
        target: "lehmer" for Lehmer extensions, "seed" for plus-seed extensions.
    """
    if not seed.is_plus_seed():
        return []
    N = seed.evaluation
    delta = lehmer_delta(N) if target == "lehmer" else seed_delta(N)
    factors = factor_fn(delta)
    pairs = divisor_pairs_from_factorization(factors, delta)

    results: list[Factorization] = []
    for d1, d2 in pairs:
        ext = seed.with_factors(N + 1 + d1, N + 1 + d2)
        if (target == "lehmer" and ext.is_lehmer()) or (
            target == "seed" and ext.is_plus_seed()
        ):
            results.append(ext)
    return results
