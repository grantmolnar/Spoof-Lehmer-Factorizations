"""Investigation of primitive k-Lehmer factorizations.

Primitives are k-Lehmer factorizations admitting no descended pair. The
central open question is whether the set of primitives is finite for each k.

This module computes structural invariants that may shed light on the
question:

- Length distribution: at what r do primitives cluster?
- Residue patterns: distribution of eps(G) mod small primes.
- Factor multiplicity: do primitives prefer repeated small factors?
- Almost-descents: how close does each primitive come to a real descent?
- Largest factor analysis: does the largest factor satisfy bounds that
  rule out descent?
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from spoof_lehmer.domain import Factorization, find_descents
from spoof_lehmer.storage import FactorizationRepository


@dataclass
class PrimitivesReport:
    k: int
    primitives: list[Factorization]
    length_distribution: dict[int, int]
    residue_distribution: dict[int, dict[int, int]]  # modulus -> {residue: count}
    multiplicity_signatures: dict[tuple[int, ...], int]
    almost_descents: dict[Factorization, list[tuple[int, int, int]]]
    # almost_descent: (a, b, |LHS - RHS|) for the closest near-miss

    @property
    def total(self) -> int:
        return len(self.primitives)


def analyze_primitives(repo: FactorizationRepository, k: int) -> PrimitivesReport:
    """Compute structural invariants for all primitives in the repository."""
    primitives: list[Factorization] = []
    for L in repo.all_lehmers(k):
        if not find_descents(L):
            primitives.append(L)
    primitives.sort(key=lambda f: (f.length, f.evaluation))

    length_dist: Counter[int] = Counter()
    for p in primitives:
        length_dist[p.length] += 1

    residue_dist: dict[int, dict[int, int]] = {}
    for modulus in (4, 8, 16, 3, 5, 7, 11):
        counts: Counter[int] = Counter()
        for p in primitives:
            counts[p.evaluation % modulus] += 1
        residue_dist[modulus] = dict(counts)

    sig_dist: Counter[tuple[int, ...]] = Counter()
    for p in primitives:
        # Multiplicity signature: sorted tuple of multiplicities
        # e.g. (5,5,5,43,5375) -> (1,1,3) sorted = (1,1,3)
        mults = Counter(p.factors)
        sig = tuple(sorted(mults.values(), reverse=True))
        sig_dist[sig] += 1

    almost: dict[Factorization, list[tuple[int, int, int]]] = {}
    for p in primitives:
        near = _find_near_descents(p)
        if near:
            almost[p] = near

    return PrimitivesReport(
        k=k,
        primitives=primitives,
        length_distribution=dict(length_dist),
        residue_distribution=residue_dist,
        multiplicity_signatures=dict(sig_dist),
        almost_descents=almost,
    )


def _find_near_descents(
    G: Factorization, top_n: int = 3
) -> list[tuple[int, int, int]]:
    """Find pairs (a, b) closest to satisfying the descent formula.

    Returns the top_n closest pairs as (a, b, |residual|) where
    residual = N(a+b-1) - ab((a-1)(b-1)+1).
    A residual of 0 means the descent formula holds (but the seed may
    not be a plus-seed).
    """
    N = G.evaluation
    near = []
    for i in range(G.length):
        for j in range(i + 1, G.length):
            a, b = G.factors[i], G.factors[j]
            lhs = N * (a + b - 1)
            rhs = a * b * ((a - 1) * (b - 1) + 1)
            near.append((a, b, abs(lhs - rhs)))
    near.sort(key=lambda t: t[2])
    return near[:top_n]


def format_primitives_report(report: PrimitivesReport) -> str:
    lines = [
        f"=== Primitives report for k = {report.k} ===",
        f"  Total primitives: {report.total}",
        "",
        "  Length distribution:",
    ]
    for r in sorted(report.length_distribution):
        lines.append(f"    r = {r}: {report.length_distribution[r]}")

    lines.append("")
    lines.append("  Multiplicity signatures (descending mults):")
    for sig, count in sorted(
        report.multiplicity_signatures.items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        lines.append(f"    {sig}: {count}")

    lines.append("")
    lines.append("  Residue distribution (eps mod m):")
    for modulus in sorted(report.residue_distribution):
        d = report.residue_distribution[modulus]
        residues = sorted(d.items())
        bits = ", ".join(f"{r}:{c}" for r, c in residues)
        lines.append(f"    mod {modulus:2d}: {bits}")

    if report.almost_descents:
        lines.append("")
        lines.append("  Closest near-descent per primitive (a, b, |residual|):")
        for p in report.primitives:
            if p in report.almost_descents:
                near = report.almost_descents[p][0]
                lines.append(
                    f"    {p.factors}: pair ({near[0]}, {near[1]}), "
                    f"residual = {near[2]:,}"
                )
    return "\n".join(lines)
