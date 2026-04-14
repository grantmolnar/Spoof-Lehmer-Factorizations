"""Fermat-prefix closedness theorem and direct enumeration.

For each s >= 1, the Fermat-s prefix
    F^{(s)} = (F_0, F_1, ..., F_{s-1}) = (3, 5, 17, 257, 65537, ...)
is a k=2-plus-seed (eps + 1 = 2 * phi). By the s = r - 2 finishing-
feasibility identity (Proposition 2 of the paper), the k=2-Lehmer
factorizations of length s + 2 extending F^{(s)} are in bijection
with divisor pairs (d, M_s/d) with d <= sqrt(M_s) of
    M_s = 2^{2^{s+1}} - 2^{2^s} - 1.
The factorization corresponding to d is
    (F_0, ..., F_{s-1}, d + 2^{2^s}, M_s/d + 2^{2^s}).

The number of such extensions is exactly tau(M_s) / 2.

This module implements direct enumeration of these completions via
factorization of M_s, sidestepping the bounds-propagation search.
For r > s + 2 the theorem alone doesn't enumerate everything (some
completions descend through non-plus-seed intermediates), but the
Fermat-(s)-cascade -- F^{(s)} -> F^{(s+1)} -> ... -- lets us hop
along the chain.
"""
from __future__ import annotations

from math import prod
from typing import Iterator

from sympy import divisors  # type: ignore[import-untyped]


def fermat_prefix(s: int) -> tuple[int, ...]:
    """Return the Fermat-s prefix F^{(s)} = (F_0, ..., F_{s-1}).

    F_i = 2^{2^i} + 1 (the Fermat numbers; not necessarily prime).
    """
    if s < 0:
        raise ValueError("s must be >= 0")
    return tuple(2 ** (2 ** i) + 1 for i in range(s))


def fermat_M(s: int) -> int:
    """Return M_s = 2^{2^{s+1}} - 2^{2^s} - 1.

    This is the integer whose divisor pairs parameterize the length-
    (s+2) k=2-Lehmer extensions of F^{(s)}.
    """
    if s < 1:
        raise ValueError("s must be >= 1 for M_s to be defined")
    return int(2 ** (2 ** (s + 1)) - 2 ** (2 ** s) - 1)


def fermat_completions(s: int) -> Iterator[tuple[int, ...]]:
    """Yield every k=2-Lehmer factorization of length s + 2 extending
    F^{(s)}, in increasing order of x_s.

    By the Fermat-prefix closedness theorem, these are in bijection
    with divisors d <= sqrt(M_s) of M_s, with completion
        (F_0, ..., F_{s-1}, d + B, M_s/d + B)
    where B = 2^{2^s}.
    """
    if s < 1:
        raise ValueError("s must be >= 1")
    F = fermat_prefix(s)
    B = 2 ** (2 ** s)
    M = fermat_M(s)
    for d in divisors(M):
        if d * d > M:
            break
        d2 = M // d
        a = d + B
        b = d2 + B
        yield F + (a, b)


def is_plus_seed(prefix: tuple[int, ...], k: int = 2) -> bool:
    """Return True iff `prefix` is a k-plus-seed.

    .. deprecated::
        This function is preserved for backward compatibility.  New
        code should import from
        :mod:`spoof_lehmer.analysis.chain` directly.
    """
    from spoof_lehmer.analysis.chain import is_plus_seed as _is_plus_seed
    return _is_plus_seed(prefix, k=k)


def chain_extend(plus_seed: tuple[int, ...]) -> tuple[int, ...]:
    """Return the canonical chain extension of a k=2 plus-seed.

    .. deprecated::
        This function is preserved for backward compatibility.  New
        code should import from
        :mod:`spoof_lehmer.analysis.chain` directly.
    """
    from spoof_lehmer.analysis.chain import chain_extend as _chain_extend
    return _chain_extend(plus_seed, k=2)


def chain(plus_seed: tuple[int, ...], length: int) -> list[tuple[int, ...]]:
    """Return the first `length` plus-seeds in the chain starting at
    `plus_seed` (inclusive of the starting plus-seed).

    .. deprecated::
        This function is preserved for backward compatibility.  New
        code should construct a :class:`spoof_lehmer.analysis.chain.Chain`
        and call :meth:`Chain.members` or :meth:`Chain.member_at_depth`.
    """
    if length < 1:
        return []
    out = [plus_seed]
    for _ in range(length - 1):
        out.append(chain_extend(out[-1]))
    return out


def verify_fermat_completion(factors: tuple[int, ...], k: int = 2) -> bool:
    """Return True iff the given factor tuple is a valid k-Lehmer.

    Used in tests and as a defensive sanity check on theoretical
    output.
    """
    eps = prod(factors)
    phi = prod(f - 1 for f in factors)
    return k * phi == eps - 1


def count_fermat_completions(s: int) -> int:
    """Return tau(M_s) / 2, the predicted number of length-(s+2)
    extensions of F^{(s)}.
    """
    M = fermat_M(s)
    # tau(M) = number of divisors. We count via divisors() which
    # internally factors and enumerates.
    return sum(1 for _ in divisors(M)) // 2
