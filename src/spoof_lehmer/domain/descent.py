"""The descent formula. Note: this function takes no `k` argument.

This is the architecture encoding Theorem (k-independence): whether a pair
{a, b} is a descended pair of a Lehmer factorization with evaluation N
depends only on N, a, and b - not on k.
"""
from __future__ import annotations
from dataclasses import dataclass
from spoof_lehmer.domain.factorization import Factorization


@dataclass(frozen=True)
class DescentPair:
    """A descended pair (a, b) and the resulting plus-seed."""
    a: int
    b: int
    seed: Factorization


def descent_holds(N: int, a: int, b: int) -> bool:
    """The descent formula: N(a + b - 1) = ab((a-1)(b-1) + 1).

    K-INDEPENDENT. If this holds AND F\\{a,b} is a plus-seed, then {a,b}
    is a descended pair of any k-Lehmer factorization F with eval N.
    """
    return N * (a + b - 1) == a * b * ((a - 1) * (b - 1) + 1)


def find_descents(G: Factorization) -> list[DescentPair]:
    """Find all descended pairs of a k-Lehmer factorization G."""
    if not G.is_lehmer():
        return []
    descents: list[DescentPair] = []
    N = G.evaluation
    r = G.length
    seen: set[tuple[int, int]] = set()
    for i in range(r):
        for j in range(i + 1, r):
            a, b = G.factors[i], G.factors[j]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            if not descent_holds(N, a, b):
                continue
            remainder = G.without(i, j)
            if remainder.is_plus_seed():
                descents.append(DescentPair(a=a, b=b, seed=remainder))
                seen.add(key)
    return descents


def is_primitive(G: Factorization) -> bool:
    """G is primitive iff it admits no descended pair."""
    return G.is_lehmer() and len(find_descents(G)) == 0
