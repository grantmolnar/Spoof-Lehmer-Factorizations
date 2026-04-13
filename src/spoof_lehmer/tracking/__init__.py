"""Search run tracking. The exhaustiveness ledger.

A `RunResult` is what every `SearchStrategy.discover()` returns: it carries
the count of newly added factorizations, the run status, and the list of
known-but-unprocessed frontier items (e.g. seeds whose Delta exceeded the
factoring backend's max_n).

A `RunRecord` is what gets persisted to the database for later auditing.
Together with the `pending_factoring` queue, these answer the question
"how far out have I found everything, and what do I still need to do?"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class RunStatus(Enum):
    """Outcome of a single discover() invocation."""
    COMPLETE = "complete"
    """Search exhausted its declared box without hitting any limit.
    The box is "everything within (max_r, max_N)" for the recurrence,
    or "all extensions of seeds whose Delta is factorable" for the cascade."""

    NODE_LIMIT_HIT = "node_limit_hit"
    """The recurrence search hit its per-(r, target) node cap and was
    truncated. The result is incomplete even within the declared box."""

    DELTA_TOO_LARGE = "delta_too_large"
    """The cascade encountered seeds whose Delta exceeded the factoring
    backend's max_n. Those seeds were skipped and recorded in the
    pending_factoring queue. Other seeds were processed normally."""

    INTERRUPTED = "interrupted"
    """User cancelled the run."""


@dataclass
class PendingFactoring:
    """A seed the cascade could not process because Delta was too large."""
    seed_factors: tuple[int, ...]
    k: int
    delta_value: int
    delta_kind: str  # "lehmer" (Delta_L) or "seed" (Delta_S)
    backend_attempted: str
    max_n_at_attempt: int


@dataclass
class RunResult:
    """What a SearchStrategy.discover() call returns.

    Replaces the previous `int` return type so callers can audit
    completeness, not just count.
    """
    added: int
    status: RunStatus
    nodes_explored: int = 0
    pending: list[PendingFactoring] = field(default_factory=list)
    notes: str = ""

    @property
    def is_complete(self) -> bool:
        return self.status == RunStatus.COMPLETE


@dataclass
class RunRecord:
    """A persisted record of a discover() invocation."""
    id: int | None
    strategy: str
    k: int
    max_r: int | None
    max_N: int | None
    node_limit: int | None
    started_at: str
    finished_at: str
    status: str
    nodes_explored: int
    factorizations_added: int
    notes: str

    @classmethod
    def started_now(
        cls,
        strategy: str,
        k: int,
        max_r: int | None = None,
        max_N: int | None = None,
        node_limit: int | None = None,
    ) -> "RunRecord":
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return cls(
            id=None,
            strategy=strategy,
            k=k,
            max_r=max_r,
            max_N=max_N,
            node_limit=node_limit,
            started_at=now,
            finished_at=now,
            status=RunStatus.COMPLETE.value,
            nodes_explored=0,
            factorizations_added=0,
            notes="",
        )

    def finish(self, result: RunResult) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.status = result.status.value
        self.nodes_explored = result.nodes_explored
        self.factorizations_added = result.added
        self.notes = result.notes

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock duration of this run, derived from started_at and
        finished_at. Returns 0.0 if either timestamp is unparseable
        (e.g. an in-progress marker whose finished_at equals started_at).
        """
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at)
        except ValueError:
            return 0.0
        return max(0.0, (end - start).total_seconds())
