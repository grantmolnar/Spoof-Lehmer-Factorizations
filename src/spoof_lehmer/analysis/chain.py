"""Plus-seed chain theory.

Implements the chain-extension theorem from the trees paper: every
$k = 2$ plus-seed $F^*$ initiates a unique infinite chain of
plus-seeds via $x = \\eps(F^*) + 2$.  The chain is rooted at a
\\emph{fresh} plus-seed (one whose length-$(s-1)$ prefix is not itself
a plus-seed).

This module exposes three primary objects:

  * :func:`is_plus_seed` --- predicate.
  * :func:`chain_extend` --- canonical chain step ($x = E + 2$).
  * :class:`Chain` --- value object representing a rooted chain;
    iterates through its members in length order.

The chain abstraction is the structural unit for organizing plus-seeds
in the census; together with the parent-pointer machinery in
:mod:`spoof_lehmer.analysis.census_organization`, it gives the full
forest decomposition.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import prod


def is_plus_seed(factors: tuple[int, ...], k: int = 2) -> bool:
    """Return True iff ``factors`` is a $k$-plus-seed.

    A $k$-plus-seed satisfies $k \\cdot \\phi^*(F) = \\eps(F) + 1$,
    equivalently $A = k P - E = 1$ in the notation of the
    finishing-feasibility identities.
    """
    if not factors:
        # Empty product: eps = 1, phi = 1, k*phi = k, eps+1 = 2. Plus-seed iff k = 2.
        return k == 2
    return k * prod(x - 1 for x in factors) == prod(factors) + 1


def chain_extend(plus_seed: tuple[int, ...], k: int = 2) -> tuple[int, ...]:
    """Return the canonical chain extension of a plus-seed.

    For a $k = 2$ plus-seed $F^*$ with $\\eps(F^*) = E$, the unique
    value of $x$ for which $(F^*, x)$ is again a plus-seed is
    $x = E + 2$.  See trees paper Theorem (chain extension).

    Raises:
        ValueError: if ``plus_seed`` is not a $k$-plus-seed.
        NotImplementedError: for $k \\ne 2$ (the canonical extension
            depends on $k$ and is not yet implemented for $k > 2$).
    """
    if k != 2:
        raise NotImplementedError("chain_extend currently only supports k = 2")
    if not is_plus_seed(plus_seed, k=k):
        raise ValueError(f"{plus_seed} is not a k = {k} plus-seed")
    eps = prod(plus_seed) if plus_seed else 1
    return plus_seed + (eps + 2,)


def chain_parent(P: tuple[int, ...], k: int = 2) -> tuple[int, ...] | None:
    """Return the chain-parent of $P$ (the unique plus-seed $Q$ with
    $P = (Q, \\eps(Q) + 2)$), or None if $P$ is itself a fresh root.

    Returns None for the empty tuple (which is the root of the Fermat
    chain when $k = 2$).
    """
    if k != 2:
        return None
    if not P:
        return None
    Q = P[:-1]
    if not is_plus_seed(Q, k=k):
        return None
    eps_Q = prod(Q) if Q else 1
    return Q if P[-1] == eps_Q + 2 else None


def chain_root(P: tuple[int, ...], k: int = 2) -> tuple[int, ...]:
    """Walk back chain-parent pointers to the root of $P$'s chain.

    The root is the fresh plus-seed initiating the chain that contains
    $P$.  For Fermat partial products this is the empty tuple
    (representing $\\emptyset$, the root of the Fermat chain).
    """
    cur = P
    while True:
        par = chain_parent(cur, k=k)
        if par is None:
            return cur
        cur = par


def is_fresh_plus_seed(P: tuple[int, ...], k: int = 2) -> bool:
    """Return True iff $P$ is a plus-seed whose chain-parent is None,
    i.e. it is the root of a chain.

    The empty tuple is treated as the (canonical) Fermat root for
    $k = 2$.
    """
    if not is_plus_seed(P, k=k):
        return False
    return chain_parent(P, k=k) is None


@dataclass(frozen=True)
class Chain:
    """A plus-seed chain rooted at a fresh plus-seed.

    A chain is uniquely determined by its root (Theorem chain
    extension exists and is unique).  The chain itself is the infinite
    sequence ``root, chain_extend(root), chain_extend(chain_extend(root)), ...``.

    This is a value object: two ``Chain`` instances are equal iff
    their roots are equal.  ``members(up_to_length)`` enumerates the
    initial segment.

    Attributes:
        root: the fresh plus-seed at which the chain is rooted.
        k: the value of $k$ (currently only $k = 2$ is supported).
        label: an optional human-readable label (e.g. "Fermat",
            "sporadic-A").  Defaults to "Fermat" for the Fermat chain
            and "sporadic" otherwise.
    """
    root: tuple[int, ...]
    k: int = 2
    label: str = field(default="")

    def __post_init__(self) -> None:
        if self.k != 2:
            raise NotImplementedError("Chain currently only supports k = 2")
        if not is_fresh_plus_seed(self.root, k=self.k):
            raise ValueError(
                f"root {self.root} is not a fresh k = {self.k} plus-seed"
            )
        if not self.label:
            # Use object.__setattr__ because dataclass is frozen.
            default = "Fermat" if self.root in ((), (3,)) else "sporadic"
            object.__setattr__(self, "label", default)

    @property
    def is_fermat(self) -> bool:
        """Whether this is the (unique) Fermat chain."""
        return self.root in ((), (3,))

    def members(self, up_to_length: int) -> list[tuple[int, ...]]:
        """Return the chain members of length $\\le$ ``up_to_length``.

        For the Fermat chain rooted at $\\emptyset$, the empty tuple
        is included as the length-0 member.
        """
        out = []
        cur = self.root
        while len(cur) <= up_to_length:
            out.append(cur)
            cur = chain_extend(cur, k=self.k)
        return out

    def iter_members(self) -> Iterator[tuple[int, ...]]:
        """Lazy infinite iteration through chain members."""
        cur = self.root
        while True:
            yield cur
            cur = chain_extend(cur, k=self.k)

    def member_at_depth(self, depth: int) -> tuple[int, ...]:
        """Return the ``depth``-th member of the chain (0-indexed at root)."""
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        cur = self.root
        for _ in range(depth):
            cur = chain_extend(cur, k=self.k)
        return cur

    def depth_of(self, P: tuple[int, ...]) -> int:
        """Return the depth of $P$ within this chain.

        Raises:
            ValueError: if $P$ is not in this chain.
        """
        depth = 0
        cur = P
        while cur != self.root:
            par = chain_parent(cur, k=self.k)
            if par is None:
                raise ValueError(f"{P} is not in chain rooted at {self.root}")
            cur = par
            depth += 1
        return depth
