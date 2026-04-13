"""Classification of sporadic plus-seeds.

A plus-seed is "Fermat" if it equals one of the partial products
F^(s) = prod_{i=0}^{s-1}(2^{2^i} + 1) = (3, 5, 17, 257, 65537, ...).

A plus-seed is "Fermat-derived" if it strictly contains a Fermat seed
as a sub-multiset.

A plus-seed is "sporadic" otherwise.

This module identifies the sporadic seeds, computes the inclusion graph
on the seed set (which seeds are sub-multisets of which), and finds
maximal chains of seeds related by the seed-to-seed extension.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from spoof_lehmer.domain import Factorization
from spoof_lehmer.storage import FactorizationRepository

# Known Fermat numbers (the only ones we need for k=2 census ranges)
FERMAT_NUMBERS = (3, 5, 17, 257, 65537, 4294967297)


def is_fermat_seed(F: Factorization) -> bool:
    """True if F is exactly a Fermat partial product (3,5,17,...)."""
    if F.k != 2:
        return False
    expected = FERMAT_NUMBERS[: F.length]
    return F.factors == expected


def contains_fermat_seed(F: Factorization) -> bool:
    """True if F's sorted factors begin with a Fermat partial product
    of length >= 1."""
    if F.k != 2 or F.length == 0:
        return False
    for s in range(1, len(FERMAT_NUMBERS) + 1):
        if F.factors[:s] == FERMAT_NUMBERS[:s] and F.length > s:
            return True
    return False


def classify_seed(F: Factorization) -> str:
    """Return 'fermat', 'fermat-derived', or 'sporadic'."""
    if F.length == 0:
        return "fermat"
    if is_fermat_seed(F):
        return "fermat"
    if contains_fermat_seed(F):
        return "fermat-derived"
    return "sporadic"


@dataclass
class SporadicReport:
    k: int
    seeds: list[Factorization]
    classifications: dict[str, list[Factorization]]
    inclusion_graph: dict[Factorization, list[Factorization]]
    # parent -> children where parent's factors are a sub-multiset of child's
    families: list[list[Factorization]]
    # Connected components in the inclusion graph

    @property
    def total(self) -> int:
        return len(self.seeds)


def analyze_sporadic_seeds(
    repo: FactorizationRepository, k: int
) -> SporadicReport:
    """Classify and group all plus-seeds in the repository."""
    seeds = sorted(repo.all_seeds(k), key=lambda f: (f.length, f.evaluation))

    classifications: dict[str, list[Factorization]] = defaultdict(list)
    for s in seeds:
        classifications[classify_seed(s)].append(s)

    # Inclusion graph: parent -> child if parent.factors is a sub-multiset
    # of child.factors AND child.length == parent.length + 2 (one extension step).
    inclusion: dict[Factorization, list[Factorization]] = defaultdict(list)
    for parent in seeds:
        for child in seeds:
            if child.length != parent.length + 2:
                continue
            if _is_sub_multiset(parent.factors, child.factors):
                inclusion[parent].append(child)

    # Connected components in the underlying undirected inclusion graph
    families = _connected_components(seeds, inclusion)

    return SporadicReport(
        k=k,
        seeds=seeds,
        classifications=dict(classifications),
        inclusion_graph=dict(inclusion),
        families=families,
    )


def _is_sub_multiset(small: tuple[int, ...], large: tuple[int, ...]) -> bool:
    """True if every element of `small` appears in `large` with at least
    the same multiplicity."""
    from collections import Counter
    small_c = Counter(small)
    large_c = Counter(large)
    return all(large_c[k] >= v for k, v in small_c.items())


def _connected_components(
    nodes: list[Factorization],
    edges: dict[Factorization, list[Factorization]],
) -> list[list[Factorization]]:
    parent = {n: n for n in nodes}

    def find(x: Factorization) -> Factorization:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Factorization, b: Factorization) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, children in edges.items():
        for b in children:
            union(a, b)

    components: dict[Factorization, list[Factorization]] = defaultdict(list)
    for n in nodes:
        components[find(n)].append(n)
    return sorted(components.values(), key=lambda c: (-len(c), c[0].evaluation))


def format_sporadic_report(report: SporadicReport) -> str:
    lines = [
        f"=== Sporadic seed classification for k = {report.k} ===",
        f"  Total plus-seeds: {report.total}",
        "",
        "  Classification:",
    ]
    for cls in ("fermat", "fermat-derived", "sporadic"):
        seeds = report.classifications.get(cls, [])
        lines.append(f"    {cls}: {len(seeds)}")

    lines.append("")
    lines.append(f"  Connected families: {len(report.families)}")
    for i, family in enumerate(report.families, 1):
        if len(family) <= 1:
            continue
        lines.append(f"    Family {i} ({len(family)} seeds):")
        for s in sorted(family, key=lambda f: f.length):
            cls = classify_seed(s)
            tag = f" [{cls}]" if cls != "fermat" else ""
            lines.append(f"      r={s.length}: {s.factors}{tag}")

    singletons = sum(1 for f in report.families if len(f) == 1)
    if singletons:
        lines.append(f"    + {singletons} singleton families")

    return "\n".join(lines)
