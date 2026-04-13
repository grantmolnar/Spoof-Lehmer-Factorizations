"""Analysis modules: census, primitives, sporadic seeds, statistics."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from spoof_lehmer.domain import Factorization, find_descents
from spoof_lehmer.storage import FactorizationRepository
from spoof_lehmer.analysis.primitives import (
    analyze_primitives,
    format_primitives_report,
    PrimitivesReport,
)
from spoof_lehmer.analysis.sporadic import (
    analyze_sporadic_seeds,
    format_sporadic_report,
    SporadicReport,
    classify_seed,
    is_fermat_seed,
)

from spoof_lehmer.analysis.coverage import (
    compute_coverage,
    format_coverage_report,
    CoverageReport,
    CoverageBox,
)

__all__ = [
    "run_census", "format_report", "CensusReport",
    "analyze_primitives", "format_primitives_report", "PrimitivesReport",
    "analyze_sporadic_seeds", "format_sporadic_report", "SporadicReport",
    "classify_seed", "is_fermat_seed",
    "compute_coverage", "format_coverage_report", "CoverageReport", "CoverageBox",
]


@dataclass
class CensusReport:
    k: int
    seeds_total: int
    lehmers_total: int
    primitives: list[Factorization]
    derived: int
    multi_descent: list[tuple[Factorization, int]]
    by_length: dict[int, dict[str, int]]  # length -> {"seeds": n, "lehmers": n, "prim": n}

    @property
    def is_forest(self) -> bool:
        return len(self.multi_descent) == 0


def run_census(repo: FactorizationRepository, k: int) -> CensusReport:
    """Analyze the current state of the repository for parameter k."""
    primitives: list[Factorization] = []
    multi: list[tuple[Factorization, int]] = []
    derived = 0
    by_length: dict[int, dict[str, int]] = defaultdict(
        lambda: {"seeds": 0, "lehmers": 0, "prim": 0}
    )

    lehmers = list(repo.all_lehmers(k))
    seeds = list(repo.all_seeds(k))

    for s in seeds:
        by_length[s.length]["seeds"] += 1

    for L in lehmers:
        by_length[L.length]["lehmers"] += 1
        descs = find_descents(L)
        if len(descs) == 0:
            primitives.append(L)
            by_length[L.length]["prim"] += 1
        elif len(descs) == 1:
            derived += 1
        else:
            multi.append((L, len(descs)))
            derived += 1

    return CensusReport(
        k=k,
        seeds_total=len(seeds),
        lehmers_total=len(lehmers),
        primitives=sorted(primitives, key=lambda f: (f.length, f.evaluation)),
        derived=derived,
        multi_descent=multi,
        by_length=dict(by_length),
    )


def format_report(report: CensusReport) -> str:
    """Human-readable summary."""
    lines = [
        f"=== Census for k = {report.k} ===",
        f"  Plus-seeds:           {report.seeds_total}",
        f"  Lehmer factorizations:{report.lehmers_total}",
        f"    Primitive:          {len(report.primitives)}",
        f"    Derived:            {report.derived}",
        f"  Multiple descents:    {len(report.multi_descent)}",
        f"  FOREST:               {report.is_forest}",
        "",
        "  Distribution by length:",
        f"  {'r':>4} {'seeds':>8} {'lehmers':>8} {'prim':>6}",
    ]
    for r in sorted(report.by_length):
        d = report.by_length[r]
        lines.append(
            f"  {r:>4} {d['seeds']:>8} {d['lehmers']:>8} {d['prim']:>6}"
        )
    if report.primitives:
        lines.append("")
        lines.append("  Primitives:")
        for p in report.primitives:
            lines.append(f"    {p.factors}  N = {p.evaluation:,}")
    return "\n".join(lines)
