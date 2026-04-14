"""Search strategies for discovering factorizations."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Protocol
from spoof_lehmer.domain import (
    Factorization, extensions_from_seed,
)
from spoof_lehmer.factoring import FactoringBackend
from spoof_lehmer.storage import FactorizationRepository, Provenance
from spoof_lehmer.tracking import (
    RunResult, RunStatus, RunRecord, PendingFactoring,
)
from spoof_lehmer.search.bounds_propagation import (
    BoundsPropagationStrategy as BoundsPropagationStrategy,
)
from spoof_lehmer.search.progress import (
    ProgressReporter as ProgressReporter,
    SilentProgressReporter as SilentProgressReporter,
    StderrProgressReporter as StderrProgressReporter,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SearchStrategy(Protocol):
    """Discovers factorizations and writes them to a repository."""
    name: str
    def discover(self, repo: FactorizationRepository) -> RunResult: ...


class RecurrenceStrategy:
    """Bottom-up enumeration via g(F.a) = (a-1)g(F) - eps(F).

    Walks the search tree: at each node g must remain a positive integer
    until the terminal step, where g drops to -1 (Lehmer) or +1 (seed).

    Returns a RunResult whose status is COMPLETE if every (r, target)
    sub-search exhausted its box, or NODE_LIMIT_HIT if any sub-search
    was truncated by node_limit.
    """
    name = "recurrence"

    def __init__(
        self,
        k: int,
        max_r: int = 7,
        max_N: int = 10**13,
        node_limit: int = 50_000_000,
    ):
        self.k = k
        self.max_r = max_r
        self.max_N = max_N
        self.node_limit = node_limit

    def discover(self, repo: FactorizationRepository) -> RunResult:
        record = RunRecord.started_now(
            self.name, self.k, self.max_r, self.max_N, self.node_limit,
        )
        added = 0
        total_nodes = 0
        any_truncated = False
        for r in range(1, self.max_r + 1):
            sub_added, sub_nodes, truncated = self._enumerate(repo, +1, r)
            added += sub_added
            total_nodes += sub_nodes
            any_truncated = any_truncated or truncated
        for r in range(2, self.max_r + 1):
            sub_added, sub_nodes, truncated = self._enumerate(repo, -1, r)
            added += sub_added
            total_nodes += sub_nodes
            any_truncated = any_truncated or truncated
        if self.k == 2:
            empty = Factorization.empty(self.k)
            if repo.add(empty, Provenance(self.name, None, _now())):
                added += 1

        result = RunResult(
            added=added,
            status=RunStatus.NODE_LIMIT_HIT if any_truncated else RunStatus.COMPLETE,
            nodes_explored=total_nodes,
        )
        record.finish(result)
        repo.record_run(record)
        return result

    def _enumerate(
        self, repo: FactorizationRepository, target_g: int, r: int,
    ) -> tuple[int, int, bool]:
        """Returns (added_count, nodes_explored, truncated)."""
        added = 0
        nodes = [0]
        truncated = [False]

        def search(factors: list[int], g: int, N: int, min_a: int, remaining: int) -> None:
            nonlocal added
            nodes[0] += 1
            if nodes[0] > self.node_limit:
                truncated[0] = True
                return
            if remaining == 0:
                if g == target_g:
                    fact = Factorization(tuple(factors), self.k)
                    if repo.add(fact, Provenance(self.name, None, _now())):
                        added += 1
                return
            if remaining == 1:
                if g > 0 and (N + target_g) % g == 0:
                    a = (N + target_g) // g + 1
                    if a >= min_a and a >= 2 and N * a <= self.max_N:
                        fact = Factorization(tuple(sorted(factors + [a])), self.k)
                        if repo.add(fact, Provenance(self.name, None, _now())):
                            added += 1
                return
            if g <= 0:
                return
            a_lo = max(min_a, (N + g) // g + 1)
            a_hi = 2 * N // g + 5
            if N > 0:
                budget = self.max_N // N
                if budget <= 0:
                    return
                a_hi = min(a_hi, int(budget ** (1.0 / remaining)) + 2)
            a_hi = min(a_hi, 10_000_000)
            for a in range(a_lo, a_hi + 1):
                g_new = (a - 1) * g - N
                if g_new < 1:
                    continue
                N_new = N * a
                if N_new > self.max_N:
                    break
                search(factors + [a], g_new, N_new, a, remaining - 1)

        search([], self.k - 1, 1, 2, r)
        return added, nodes[0], truncated[0]


class CascadeStrategy:
    """Top-down: from each known plus-seed, factor Delta to find extensions.

    Iterates: each round both Lehmer and seed-to-seed extensions are taken,
    and newly discovered seeds become parents for the next round. The loop
    halts as soon as a round produces no new seeds.

    `max_rounds=None` (default) means "run until natural termination."
    Pass an integer to cap the number of rounds for predictable wall time.
    """
    name = "cascade"

    def __init__(
        self,
        k: int,
        backend: FactoringBackend,
        max_rounds: int | None = None,
        max_delta: int | None = None,
    ):
        self.k = k
        self.backend = backend
        self.max_rounds = max_rounds
        self.max_delta = max_delta or backend.max_n

    def discover(self, repo: FactorizationRepository) -> RunResult:
        record = RunRecord.started_now(self.name, self.k)
        added_total = 0
        any_skipped = False
        frontier = list(repo.all_seeds(self.k))
        round_num = 0
        from spoof_lehmer.domain.extension import lehmer_delta, seed_delta
        while frontier:
            round_num += 1
            if self.max_rounds is not None and round_num > self.max_rounds:
                break
            new_seeds: list[Factorization] = []
            for seed in frontier:
                N = seed.evaluation
                # Check both Delta values; record pending if too large
                for delta_kind, delta_fn in (("lehmer", lehmer_delta), ("seed", seed_delta)):
                    delta = delta_fn(N)
                    if delta > self.max_delta:
                        any_skipped = True
                        pending = PendingFactoring(
                            seed_factors=seed.factors,
                            k=self.k,
                            delta_value=delta,
                            delta_kind=delta_kind,
                            backend_attempted=self.backend.name,
                            max_n_at_attempt=self.max_delta,
                        )
                        repo.add_pending(pending)
                        continue
                    for ext in extensions_from_seed(seed, self.backend.factor, target=delta_kind):
                        prov = Provenance(
                            f"{self.name}-{delta_kind}", seed.factors, _now()
                        )
                        if repo.add(ext, prov):
                            added_total += 1
                            if delta_kind == "seed":
                                new_seeds.append(ext)
            if not new_seeds:
                break
            frontier = new_seeds

        result = RunResult(
            added=added_total,
            status=RunStatus.DELTA_TOO_LARGE if any_skipped else RunStatus.COMPLETE,
        )
        record.finish(result)
        repo.record_run(record)
        return result

    def _safe_extensions(
        self, seed: Factorization, target: str
    ) -> list[Factorization]:
        """Legacy entry point retained for tests; main loop uses extensions_from_seed directly."""
        from spoof_lehmer.domain.extension import lehmer_delta, seed_delta
        N = seed.evaluation
        delta = lehmer_delta(N) if target == "lehmer" else seed_delta(N)
        if delta > self.max_delta:
            return []
        return extensions_from_seed(seed, self.backend.factor, target=target)
