"""Hierarchical organization of the spoof Lehmer census.

Given a set of $k$-Lehmer factorizations, this module classifies each
by its parentage relationship to the plus-seed forest:

  * ``HASANALIZADE_DESCENDED`` --- $F[:-2]$ is a plus-seed; $F$ arises
    from the standard Hasanalizade two-factor descent.
  * ``LEHMER_COMPANION`` --- $F[:-1]$ is a plus-seed $P^*$ and
    $F[-1] = \\eps(P^*)$, i.e. $F$ is the Lehmer companion of a
    plus-seed via the chain remark "two children per plus-seed".
  * ``PRIMITIVE`` --- neither: $F$ has no plus-seed parent of either
    kind.
  * ``TRIVIAL`` --- length-2 case $(3, 3)$, the unique smallest
    Lehmer factorization.

Plus-seeds are then themselves grouped by chain root via
:mod:`spoof_lehmer.analysis.chain`.

This module is pure: it operates on tuples and produces dataclasses,
with no I/O.  The :mod:`spoof_lehmer.reporting` layer is responsible
for rendering (LaTeX, JSON, plain text).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from spoof_lehmer.analysis.chain import (
    Chain,
    chain_parent,
    chain_root,
    is_fresh_plus_seed,
    is_plus_seed,
)


class ParentKind(str, Enum):
    """The kind of parent relationship a Lehmer factorization has."""
    HASANALIZADE_DESCENDED = "descended"
    LEHMER_COMPANION = "companion"
    PRIMITIVE = "primitive"
    TRIVIAL = "trivial"


@dataclass(frozen=True)
class FactorizationProvenance:
    """Provenance attribution for a single Lehmer factorization.

    Attributes:
        factors: the Lehmer factorization itself.
        kind: the category (descended, companion, primitive, trivial).
        parent: the plus-seed parent if applicable, else None.
            For ``HASANALIZADE_DESCENDED`` it is ``factors[:-2]``;
            for ``LEHMER_COMPANION`` it is ``factors[:-1]``;
            for ``PRIMITIVE`` and ``TRIVIAL`` it is None.
    """
    factors: tuple[int, ...]
    kind: ParentKind
    parent: tuple[int, ...] | None


def classify(factors: tuple[int, ...], k: int = 2) -> FactorizationProvenance:
    """Classify a single Lehmer factorization by its parentage."""
    if len(factors) < 3:
        return FactorizationProvenance(factors, ParentKind.TRIVIAL, None)
    if is_plus_seed(factors[:-2], k=k):
        return FactorizationProvenance(
            factors, ParentKind.HASANALIZADE_DESCENDED, factors[:-2],
        )
    if is_plus_seed(factors[:-1], k=k):
        return FactorizationProvenance(
            factors, ParentKind.LEHMER_COMPANION, factors[:-1],
        )
    return FactorizationProvenance(factors, ParentKind.PRIMITIVE, None)


@dataclass(frozen=True)
class CensusOrganization:
    """Result of organizing a census by chain forest structure.

    Attributes:
        provenances: classification of every input Lehmer
            factorization, in the order given.
        descended_to_children: map from each plus-seed parent to its
            Hasanalizade-descended children (sorted by $x_{r-1}$,
            i.e. the second-to-last factor).
        companion_to_child: map from each plus-seed $P^*$ to its
            unique Lehmer companion $(P^*, \\eps(P^*))$, when that
            companion is in the input census.
        primitives_by_length: map from length $r$ to the truly
            primitive Lehmer factorizations of that length.
        chains: every plus-seed chain whose members appear in the
            census, keyed by chain root.

    The invariant ``len(provenances) == sum of all sub-categories``
    holds.
    """
    provenances: tuple[FactorizationProvenance, ...]
    descended_to_children: dict[tuple[int, ...], list[tuple[int, ...]]]
    companion_to_child: dict[tuple[int, ...], tuple[int, ...]]
    primitives_by_length: dict[int, list[tuple[int, ...]]]
    chains: dict[tuple[int, ...], Chain]

    def chain_for(self, plus_seed: tuple[int, ...]) -> Chain:
        """Return the chain containing the given plus-seed."""
        return self.chains[chain_root(plus_seed)]

    def fresh_plus_seeds_at_length(self, length: int) -> list[tuple[int, ...]]:
        """Return all fresh plus-seeds (chain roots) of the given length
        observed in the census.

        A fresh plus-seed is a plus-seed whose chain-parent is None, i.e.
        whose length-$(s-1)$ prefix is not itself a plus-seed.

        This method is the structural form of the "inverse parent
        inference" technique described in the paper: fresh length-$s$
        plus-seeds are discoverable from the census without an
        explicit length-$s$ plus-seed search.  Each appears as either
        the parent of a Hasanalizade-descended Lehmer factorization
        (when $r = s + 2$) or as the parent of a Lehmer-companion
        factorization (when $r = s + 1$).
        """
        return sorted(
            chain.root for chain in self.chains.values()
            if len(chain.root) == length
        )

    def descendants_of_chain(self, chain: Chain) -> int:
        """Return the total number of Lehmer factorizations descending
        from any member of the given chain.

        Counts both Hasanalizade-descended children and Lehmer
        companions.
        """
        parents_in_census = (
            set(self.descended_to_children) | set(self.companion_to_child)
        )
        # Walk chain only up to the longest parent length in the census;
        # beyond that, no descendants can exist anyway.
        max_length = max(
            (len(p) for p in parents_in_census),
            default=0,
        )
        members_in_census = set(chain.members(up_to_length=max_length)) & parents_in_census

        n = 0
        for P in members_in_census:
            n += len(self.descended_to_children.get(P, []))
            if P in self.companion_to_child:
                n += 1
        return n


def organize(
    census: Iterable[Sequence[int]],
    k: int = 2,
) -> CensusOrganization:
    """Organize a $k$-Lehmer census by chain forest structure.

    Args:
        census: iterable of Lehmer factorizations (each a sequence of
            integer factors).  Only entries with the given $k$ are
            considered.
        k: the value of $k$ (currently only $k = 2$ is supported).

    Returns:
        A :class:`CensusOrganization` containing the full
        classification.
    """
    if k != 2:
        raise NotImplementedError("organize currently only supports k = 2")

    provenances: list[FactorizationProvenance] = []
    descended: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
    companion: dict[tuple[int, ...], tuple[int, ...]] = {}
    primitives: dict[int, list[tuple[int, ...]]] = defaultdict(list)

    for entry in census:
        F = tuple(entry)
        prov = classify(F, k=k)
        provenances.append(prov)
        if prov.kind is ParentKind.HASANALIZADE_DESCENDED:
            assert prov.parent is not None
            descended[prov.parent].append(F)
        elif prov.kind is ParentKind.LEHMER_COMPANION:
            assert prov.parent is not None
            companion[prov.parent] = F
        elif prov.kind is ParentKind.PRIMITIVE:
            primitives[len(F)].append(F)

    # Sort descended children by the second-to-last factor (x_{r-1}),
    # which is the most natural ordering within a parent group.
    for parent, kids in descended.items():
        kids.sort(key=lambda f: f[-2])

    # Build the chain dictionary: include every plus-seed appearing as
    # a parent, and walk back to its chain root.
    chains: dict[tuple[int, ...], Chain] = {}
    parent_seeds = set(descended) | set(companion)
    closure: set[tuple[int, ...]] = set()
    for P in parent_seeds:
        cur: tuple[int, ...] | None = P
        while cur is not None and cur not in closure:
            closure.add(cur)
            cur = chain_parent(cur, k=k)

    for P in closure:
        root = chain_root(P, k=k)
        if root not in chains:
            # Only build a Chain object if root is actually a fresh plus-seed.
            if is_fresh_plus_seed(root, k=k):
                chains[root] = Chain(root=root, k=k)

    return CensusOrganization(
        provenances=tuple(provenances),
        descended_to_children=dict(descended),
        companion_to_child=companion,
        primitives_by_length=dict(primitives),
        chains=chains,
    )
