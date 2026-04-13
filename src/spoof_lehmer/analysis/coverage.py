"""Exhaustiveness queries against the search-run ledger.

For each length r, computes the largest box (r, max_N) for which the
database is provably exhaustive, by examining the recurrence runs in the
search_runs table. Also reports the cascade pending queue.

The output of this module answers the question:
    "How far out have I found everything, and what do I need to do next?"
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from spoof_lehmer.storage import FactorizationRepository
from spoof_lehmer.tracking import RunStatus


@dataclass
class CoverageBox:
    """Exhaustiveness status for a single length r."""
    r: int
    max_N_complete: int        # largest max_N from a COMPLETE recurrence run
    truncated_runs: int         # number of runs at this r that hit node_limit
    has_complete_run: bool

    @property
    def status(self) -> str:
        if self.has_complete_run and self.truncated_runs == 0:
            return "EXHAUSTIVE"
        if self.has_complete_run:
            return "EXHAUSTIVE_BELOW_TRUNCATION"
        if self.truncated_runs > 0:
            return "TRUNCATED"
        return "NEVER_RUN"


@dataclass
class CoverageReport:
    k: int
    boxes: dict[int, CoverageBox]      # r -> CoverageBox
    pending_count: int
    pending_by_kind: dict[str, int]    # delta_kind -> count
    smallest_unprocessed_delta: int | None


def compute_coverage(repo: FactorizationRepository, k: int, max_r: int = 10) -> CoverageReport:
    """Inspect the search-run ledger and produce a coverage report."""
    runs = repo.all_runs(k)

    # For each r, find the largest max_N from a complete recurrence run
    by_r_complete: dict[int, int] = defaultdict(int)
    by_r_truncated: dict[int, int] = defaultdict(int)
    by_r_has_complete: dict[int, bool] = defaultdict(bool)

    for run in runs:
        if run.strategy != "recurrence":
            continue
        if run.max_r is None or run.max_N is None:
            continue
        # A recurrence run with max_r=R covers every r in [1, R]
        for r in range(1, run.max_r + 1):
            if run.status == RunStatus.COMPLETE.value:
                by_r_has_complete[r] = True
                if run.max_N > by_r_complete[r]:
                    by_r_complete[r] = run.max_N
            elif run.status == RunStatus.NODE_LIMIT_HIT.value:
                by_r_truncated[r] += 1

    boxes = {}
    for r in range(1, max_r + 1):
        boxes[r] = CoverageBox(
            r=r,
            max_N_complete=by_r_complete.get(r, 0),
            truncated_runs=by_r_truncated.get(r, 0),
            has_complete_run=by_r_has_complete.get(r, False),
        )

    # Pending queue summary
    pending = repo.all_pending(k) if hasattr(repo, "all_pending") else []
    pending_by_kind: dict[str, int] = defaultdict(int)
    smallest_delta: int | None = None
    for p in pending:
        pending_by_kind[p.delta_kind] += 1
        if smallest_delta is None or p.delta_value < smallest_delta:
            smallest_delta = p.delta_value

    return CoverageReport(
        k=k,
        boxes=boxes,
        pending_count=len(pending),
        pending_by_kind=dict(pending_by_kind),
        smallest_unprocessed_delta=smallest_delta,
    )


def format_coverage_report(report: CoverageReport) -> str:
    lines = [
        f"=== Exhaustiveness coverage for k = {report.k} ===",
        "",
        f"  {'r':>3}  {'status':<32}  {'max_N (complete)':>22}",
    ]
    for r in sorted(report.boxes):
        box = report.boxes[r]
        if box.max_N_complete > 0:
            n_str = f"{box.max_N_complete:.2e}"
        else:
            n_str = "-"
        lines.append(f"  {r:>3}  {box.status:<32}  {n_str:>22}")

    lines.append("")
    lines.append("  Cascade pending queue (Delta too large for backend):")
    if report.pending_count == 0:
        lines.append("    (empty - cascade processed every known seed)")
    else:
        lines.append(f"    Total: {report.pending_count}")
        for kind, count in sorted(report.pending_by_kind.items()):
            lines.append(f"      {kind}: {count}")
        if report.smallest_unprocessed_delta is not None:
            digits = len(str(report.smallest_unprocessed_delta))
            lines.append(
                f"    Smallest unprocessed Delta: ~10^{digits-1} "
                f"({digits} digits)"
            )

    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    EXHAUSTIVE                   - every factorization in box (r, max_N) is in DB")
    lines.append("    EXHAUSTIVE_BELOW_TRUNCATION  - complete run exists, but a larger run was truncated")
    lines.append("    TRUNCATED                    - all runs at this r hit node_limit; coverage unknown")
    lines.append("    NEVER_RUN                    - no recurrence run has covered this r")
    return "\n".join(lines)
