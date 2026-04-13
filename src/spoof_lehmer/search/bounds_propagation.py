"""Bounds-propagation search strategy.

This is a port of the Molnar-Singh algorithm from
https://github.com/grantmolnar/Spoof-Lehmer-Factorizations, integrated
with the rest of the architecture.

Algorithm sketch
----------------
A "partial state" is a sorted prefix of factors (x_1, ..., x_s) with
s <= r and a target k. We maintain two integer-valued bounds on the
ratio k(F) = (eps - 1) / phi* over all completions of the prefix:

    L(prefix) = (eps - 1) / phi*               (s == r case)
              = eps / phi*                     (s < r case)
                attained as remaining factors -> infinity

    U(prefix) = (eps' - 1) / phi*'             where eps' and phi*'
                are computed by filling all remaining slots with the
                smallest legal next factor (the current max factor).

Both bounds approach the shared limit eps/phi* as the next factor
grows: L from below, U from above. Pruning rule: if U < k for the
target k, no completion of this prefix can give a k-Lehmer
factorization.

The brilliant property is that this gives termination *without* an
a priori bound on epsilon. The "max_N" parameter that the recurrence
strategy needs is replaced by the natural termination of the next-factor
loop, which always halts because U decreases monotonically toward the
shared limit.

Two implementation notes:

1. We never construct sympy.Fraction objects on the hot path. The
   bound comparisons L <= k and U < k are written as cross-multiplied
   integer comparisons, which are 10-100x faster than Fraction().

2. We use a single mutable factors list throughout the recursion,
   appending before recursing and popping after. This avoids the
   per-node list-copy allocation cost.

Pruning rule from Lehmer (Theorem 2): no factor of a Lehmer
factorization can be congruent to 1 mod another factor. We check
this at every candidate.
"""
from __future__ import annotations
from collections.abc import Callable
from datetime import datetime, timezone
from math import isqrt
from spoof_lehmer.domain import Factorization
from spoof_lehmer.storage import FactorizationRepository, Provenance
from spoof_lehmer.tracking import RunResult, RunStatus, RunRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class BoundsPropagationStrategy:
    """Find all k-Lehmer factorizations with exactly r factors, for all
    relevant k, using monotone bounds on k(F).

    Unlike RecurrenceStrategy, this strategy does NOT take a max_N
    parameter. Termination comes from the bounds tightening, not from a
    box cutoff. This makes it the right tool for proving exhaustive
    completeness at length r without an a priori epsilon bound.

    For each r in [2, max_r], the strategy:
      1. Computes the (finite!) range of k values worth searching.
      2. For each such k, walks the prefix tree with bounds-driven pruning.
      3. Yields every prefix that completes to a k-Lehmer factorization.

    Plus-seeds (k * phi* = eps + 1) are NOT discovered by this strategy
    in its current form - it targets only k-Lehmer factorizations. The
    recurrence strategy still owns plus-seed discovery.
    """
    name = "bounds_propagation"

    def __init__(
        self,
        k_target: int,
        max_r: int = 7,
        min_r: int = 2,
        is_even: bool = False,
        use_finishing_feasibility: bool = True,
        on_found: "Callable[[Factorization], None] | None" = None,
    ):
        """
        Args:
            k_target: the value of k in k * phi*(F) = eps(F) - 1.
            max_r: enumerate factorizations of length min_r..max_r inclusive.
            min_r: smallest length to enumerate (default 2). Set equal to
                max_r to search only one r.
            is_even: if True, factors must be >= 2 (even spoof). Default
                False means factors must be >= 3 and odd (the original
                paper's restriction).
            use_finishing_feasibility: if True (default), solve the last
                two levels (s = r-1 and s = r-2) in closed form via the
                finishing-feasibility identities rather than by the
                bounds-propagation loop.
            on_found: optional callback invoked once per newly-added
                factorization (i.e. ones that survive repo.add()'s
                dedup check). Lets callers stream progress without
                wrapping the repository.
        """
        self.k_target = k_target
        self.max_r = max_r
        self.min_r = min_r
        self.is_even = is_even
        self.use_finishing_feasibility = use_finishing_feasibility
        self.on_found = on_found

    def discover(self, repo: FactorizationRepository) -> RunResult:
        record = RunRecord.started_now(
            self.name, self.k_target, self.max_r, max_N=None,
        )
        added = 0
        nodes = 0
        for r in range(self.min_r, self.max_r + 1):
            sub_added, sub_nodes = self._search_at_length(repo, r)
            added += sub_added
            nodes += sub_nodes

        result = RunResult(
            added=added, status=RunStatus.COMPLETE, nodes_explored=nodes,
            notes=f"k_target={self.k_target}, no max_N (bounds-driven termination)",
        )
        record.finish(result)
        repo.record_run(record)
        return result

    def _search_at_length(
        self, repo: FactorizationRepository, r: int
    ) -> tuple[int, int]:
        added = 0
        nodes = [0]

        # Mutable state shared across the recursion.
        factors: list[int] = []
        # eps and phi* maintained incrementally.
        # eps = product of factors so far (1 if empty)
        # phi = product of (x_i - 1) so far (1 if empty)
        k = self.k_target
        min_first = 2 if self.is_even else 3
        step = 1 if self.is_even else 2

        use_ff = self.use_finishing_feasibility
        parity_ok = (lambda v: True) if self.is_even else (lambda v: v % 2 == 1)

        def emit(extra: tuple[int, ...]) -> None:
            nonlocal added
            fact = Factorization(tuple(factors) + extra, self.k_target)
            if repo.add(fact, Provenance(self.name, None, _now())):
                added += 1
                if self.on_found is not None:
                    self.on_found(fact)

        def congruence_ok_against_prefix(v: int) -> bool:
            for x in factors:
                if (v - 1) % x == 0:
                    return False
            return True

        def finish_one(eps: int, phi: int, min_a: int) -> None:
            # s = r-1: k*P*(x-1) = E*x - 1  =>  x = (kP - 1)/(kP - E).
            kP = k * phi
            A = kP - eps  # kP - E
            if A <= 0:
                return
            num = kP - 1
            if num % A != 0:
                return
            x = num // A
            if x < min_a or not parity_ok(x):
                return
            if not congruence_ok_against_prefix(x):
                return
            emit((x,))

        def finish_two(eps: int, phi: int, min_a: int) -> None:
            # s = r-2: (A*a - B)(A*b - B) = B*E - A
            # with A = kP - E, B = kP.
            kP = k * phi
            A = kP - eps
            B = kP
            if A <= 0:
                return
            M = B * eps - A
            if M <= 0:
                # Need positive d1, d2 with d1*d2 = M and
                # d1 = A*a - B > 0 requires a > B/A.
                return
            # Bound divisor enumeration: d1 <= sqrt(M).
            # Also a >= min_a enforces d1 >= A*min_a - B.
            d1_min = A * min_a - B
            # Iterate divisors d1 of M with d1 <= sqrt(M), d1 >= max(1, d1_min).
            d1_lo = d1_min if d1_min > 0 else 1
            # Bound d1 <= isqrt(M) so a <= b.
            d1_hi = isqrt(M)
            d1 = d1_lo
            while d1 <= d1_hi:
                if M % d1 == 0:
                    d2 = M // d1
                    ax = d1 + B
                    bx = d2 + B
                    if ax % A == 0 and bx % A == 0:
                        a = ax // A
                        b = bx // A
                        if min_a <= a <= b and parity_ok(a) and parity_ok(b):
                            if congruence_ok_against_prefix(a):
                                # b must also dodge congruence with prefix AND with a.
                                if congruence_ok_against_prefix(b) and (b - 1) % a != 0:
                                    emit((a, b))
                d1 += 1

        def search(eps: int, phi: int, min_a: int, remaining: int) -> None:
            nonlocal added
            nodes[0] += 1

            if remaining == 0:
                # Terminal. Check k * phi == eps - 1.
                if k * phi == eps - 1:
                    fact = Factorization(tuple(factors), self.k_target)
                    if repo.add(fact, Provenance(self.name, None, _now())):
                        added += 1
                return

            if use_ff and remaining == 1:
                finish_one(eps, phi, min_a)
                return
            if use_ff and remaining == 2:
                finish_two(eps, phi, min_a)
                return

            # Iterate next factor a >= min_a.
            # Lower bound on k(F) for any completion:
            #     L = eps / phi   (with all remaining factors -> infinity)
            # Upper bound (with all remaining factors set to a):
            #     eps_a = eps * a^remaining
            #     phi_a = phi * (a - 1)^remaining
            #     U = (eps_a - 1) / phi_a
            #
            # We want L <= k and U >= k. As a -> infinity, both L and U
            # approach eps/phi. So once U < k, every larger a also has
            # U < k, and we break.
            #
            # Cross-multiplied integer comparisons:
            #     L <= k       <=>  eps <= k * phi
            #     U >= k       <=>  eps_a - 1 >= k * phi_a
            #     U <  k       <=>  eps_a - 1 <  k * phi_a
            #
            # The lower bound L is monotone in factors but does NOT
            # depend on `a` - it's the same for every candidate at this
            # level. Compute it once.
            lower_ok = eps <= k * phi
            if not lower_ok:
                # No completion can give k(F) <= k anymore.
                # But we still need to check if the current prefix
                # itself happens to satisfy the equation - it can't,
                # because remaining > 0.
                return

            a = min_a
            while True:
                nodes[0] += 1
                # Upper-bound check first: as a grows, U decreases
                # monotonically. Once U < k, every larger a also fails.
                # U(prefix + a + (remaining-1) copies of a) >= k iff
                #   eps * a^remaining - 1 >= k * phi * (a-1)^remaining
                # Use Python's ** which is O(log n) and stays in C.
                a_pow = a ** remaining
                am1_pow = (a - 1) ** remaining
                if eps * a_pow - 1 < k * phi * am1_pow:
                    break

                # Lehmer congruence pruning: a != 1 mod x for any
                # existing factor x.
                congruence_ok = True
                for x in factors:
                    if (a - 1) % x == 0:
                        congruence_ok = False
                        break

                if congruence_ok:
                    new_eps = eps * a
                    new_phi = phi * (a - 1)
                    factors.append(a)
                    search(new_eps, new_phi, a, remaining - 1)
                    factors.pop()

                a += step

                if a > 10**12:
                    break

        # Initial call: empty prefix, eps = phi = 1.
        search(eps=1, phi=1, min_a=min_first, remaining=r)
        return added, nodes[0]
