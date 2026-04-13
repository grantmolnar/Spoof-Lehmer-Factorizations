"""Persistence layer for factorizations and their provenance."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, TYPE_CHECKING
from spoof_lehmer.domain import Factorization

if TYPE_CHECKING:
    from spoof_lehmer.tracking import PendingFactoring, RunRecord


@dataclass(frozen=True)
class Provenance:
    """How a factorization was discovered."""
    strategy: str           # e.g. "recurrence", "cascade-lehmer", "cascade-seed"
    parent_seed: tuple[int, ...] | None = None  # for cascade extensions
    discovered_at: str = ""  # ISO timestamp


class FactorizationWriter(Protocol):
    """Minimal write-side Protocol used by SearchStrategy implementations.
    Strategies depend on this, not on the wider repository surface, so
    they stay honest about what state they touch.
    """

    def add(self, fact: Factorization, provenance: Provenance) -> bool:
        """Insert if new. Return True if newly added."""
        ...


class RunLedger(Protocol):
    """Run-tracking Protocol. Separate from FactorizationWriter so the
    interrupt-safety flow (begin_run / mark_run_complete / abandon_run)
    is uniformly available across storage backends, and so strategies
    that don't record runs needn't depend on it.
    """

    def begin_run(self, strategy: str, k: int, max_r: int) -> int:
        """Write an 'interrupted' (in-progress) search_runs row and
        return its id. If the process dies before mark_run_complete
        is called, the row survives and signals the next run to
        redo this (strategy, k, max_r) triple.
        """
        ...

    def mark_run_complete(self, run_id: int, record: "RunRecord") -> None:
        """Upgrade an in-progress run to status=complete and persist
        the RunResult statistics. Called after a clean discover()."""
        ...

    def abandon_run(self, run_id: int) -> None:
        """Delete an in-progress row. Called when a strategy's
        discover() returned cleanly but we want to replace the marker
        with the strategy's own more detailed COMPLETE row."""
        ...

    def record_run(self, record: "RunRecord") -> int:
        """Write a RunRecord directly. Used by strategies that manage
        their own completion bookkeeping and don't need the
        begin/mark/abandon three-step."""
        ...

    def all_runs(self, k: int) -> list["RunRecord"]: ...

    def add_pending(self, item: "PendingFactoring") -> bool: ...

    def all_pending(self, k: int) -> list["PendingFactoring"]: ...


class FactorizationReader(Protocol):
    """Read-side Protocol used by analysis and reporting code."""

    def contains(self, fact: Factorization) -> bool: ...

    def all_lehmers(self, k: int) -> Iterator[Factorization]: ...

    def all_seeds(self, k: int) -> Iterator[Factorization]: ...

    def by_length(self, k: int, length: int) -> Iterator[Factorization]: ...

    def count_by_kind(self, k: int) -> dict[str, int]: ...


class FactorizationRepository(
    FactorizationWriter, FactorizationReader, RunLedger, Protocol
):
    """The full repository surface. Concrete backends implement this;
    callers should prefer the narrow Protocols above when possible."""

    def close(self) -> None: ...


class SQLiteRepository:
    """SQLite-backed repository. Resumable across runs."""

    def __init__(self, path: Path | str = "data/census.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS factorizations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                k           INTEGER NOT NULL,
                kind        TEXT NOT NULL,
                length      INTEGER NOT NULL,
                evaluation  TEXT NOT NULL,    -- str(int) for big-int
                factors     TEXT NOT NULL,    -- JSON list
                strategy    TEXT NOT NULL,
                parent_seed TEXT,             -- JSON list or NULL
                discovered_at TEXT,
                UNIQUE(k, factors)
            );
            CREATE INDEX IF NOT EXISTS idx_kind_k ON factorizations(k, kind);
            CREATE INDEX IF NOT EXISTS idx_length_k ON factorizations(k, length);

            CREATE TABLE IF NOT EXISTS search_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy    TEXT NOT NULL,
                k           INTEGER NOT NULL,
                max_r       INTEGER,
                max_N       TEXT,             -- str(int)
                node_limit  INTEGER,
                started_at  TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                status      TEXT NOT NULL,
                nodes_explored INTEGER NOT NULL DEFAULT 0,
                factorizations_added INTEGER NOT NULL DEFAULT 0,
                notes       TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_runs_k ON search_runs(k);

            CREATE TABLE IF NOT EXISTS pending_factoring (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                k           INTEGER NOT NULL,
                seed_factors TEXT NOT NULL,
                delta_value TEXT NOT NULL,
                delta_kind  TEXT NOT NULL,
                backend_attempted TEXT NOT NULL,
                max_n_at_attempt TEXT NOT NULL,
                added_at    TEXT NOT NULL,
                UNIQUE(k, seed_factors, delta_kind)
            );
            CREATE INDEX IF NOT EXISTS idx_pending_k ON pending_factoring(k);
        """)
        self.conn.commit()

    def record_run(self, record: "RunRecord") -> int:
        cur = self.conn.execute(
            "INSERT INTO search_runs (strategy, k, max_r, max_N, node_limit, "
            "started_at, finished_at, status, nodes_explored, "
            "factorizations_added, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.strategy, record.k, record.max_r,
                str(record.max_N) if record.max_N is not None else None,
                record.node_limit,
                record.started_at, record.finished_at, record.status,
                record.nodes_explored, record.factorizations_added, record.notes,
            ),
        )
        self.conn.commit()
        return cur.lastrowid or 0

    def begin_run(self, strategy: str, k: int, max_r: int) -> int:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        cur = self.conn.execute(
            "INSERT INTO search_runs (strategy, k, max_r, max_N, node_limit, "
            "started_at, finished_at, status, nodes_explored, "
            "factorizations_added, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                strategy, k, max_r, None, None,
                now, now, "interrupted", 0, 0, "in-progress",
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def mark_run_complete(self, run_id: int, record: "RunRecord") -> None:
        if run_id <= 0:
            return
        self.conn.execute(
            "UPDATE search_runs SET finished_at = ?, status = ?, "
            "nodes_explored = ?, factorizations_added = ?, notes = ? "
            "WHERE id = ?",
            (
                record.finished_at, record.status,
                record.nodes_explored, record.factorizations_added,
                record.notes, run_id,
            ),
        )
        self.conn.commit()

    def abandon_run(self, run_id: int) -> None:
        if run_id <= 0:
            return
        self.conn.execute("DELETE FROM search_runs WHERE id = ?", (run_id,))
        self.conn.commit()

    def all_runs(self, k: int) -> list["RunRecord"]:
        from spoof_lehmer.tracking import RunRecord
        cur = self.conn.execute(
            "SELECT id, strategy, k, max_r, max_N, node_limit, started_at, "
            "finished_at, status, nodes_explored, factorizations_added, notes "
            "FROM search_runs WHERE k = ? ORDER BY id",
            (k,),
        )
        return [
            RunRecord(
                id=row[0], strategy=row[1], k=row[2], max_r=row[3],
                max_N=int(row[4]) if row[4] else None,
                node_limit=row[5],
                started_at=row[6], finished_at=row[7], status=row[8],
                nodes_explored=row[9], factorizations_added=row[10], notes=row[11],
            )
            for row in cur
        ]

    def add_pending(self, item: "PendingFactoring") -> bool:
        from datetime import datetime, timezone
        try:
            self.conn.execute(
                "INSERT INTO pending_factoring (k, seed_factors, delta_value, "
                "delta_kind, backend_attempted, max_n_at_attempt, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item.k, json.dumps(list(item.seed_factors)),
                    str(item.delta_value), item.delta_kind,
                    item.backend_attempted, str(item.max_n_at_attempt),
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def all_pending(self, k: int) -> list["PendingFactoring"]:
        from spoof_lehmer.tracking import PendingFactoring
        cur = self.conn.execute(
            "SELECT k, seed_factors, delta_value, delta_kind, "
            "backend_attempted, max_n_at_attempt FROM pending_factoring "
            "WHERE k = ?",
            (k,),
        )
        return [
            PendingFactoring(
                k=row[0],
                seed_factors=tuple(json.loads(row[1])),
                delta_value=int(row[2]),
                delta_kind=row[3],
                backend_attempted=row[4],
                max_n_at_attempt=int(row[5]),
            )
            for row in cur
        ]

    def add(self, fact: Factorization, provenance: Provenance) -> bool:
        try:
            self.conn.execute(
                "INSERT INTO factorizations (k, kind, length, evaluation, factors, "
                "strategy, parent_seed, discovered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact.k,
                    fact.kind.value,
                    fact.length,
                    str(fact.evaluation),
                    json.dumps(list(fact.factors)),
                    provenance.strategy,
                    json.dumps(list(provenance.parent_seed)) if provenance.parent_seed else None,
                    provenance.discovered_at,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def contains(self, fact: Factorization) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM factorizations WHERE k = ? AND factors = ?",
            (fact.k, json.dumps(list(fact.factors))),
        )
        return cur.fetchone() is not None

    def _row_to_fact(self, k: int, factors_json: str) -> Factorization:
        return Factorization(factors=tuple(json.loads(factors_json)), k=k)

    def all_lehmers(self, k: int) -> Iterator[Factorization]:
        cur = self.conn.execute(
            "SELECT factors FROM factorizations WHERE k = ? AND kind = 'lehmer' "
            "ORDER BY length, evaluation",
            (k,),
        )
        for (factors_json,) in cur:
            yield self._row_to_fact(k, factors_json)

    def all_seeds(self, k: int) -> Iterator[Factorization]:
        cur = self.conn.execute(
            "SELECT factors FROM factorizations WHERE k = ? AND kind = 'plus_seed' "
            "ORDER BY length, evaluation",
            (k,),
        )
        for (factors_json,) in cur:
            yield self._row_to_fact(k, factors_json)

    def by_length(self, k: int, length: int) -> Iterator[Factorization]:
        cur = self.conn.execute(
            "SELECT factors FROM factorizations WHERE k = ? AND length = ? "
            "ORDER BY evaluation",
            (k, length),
        )
        for (factors_json,) in cur:
            yield self._row_to_fact(k, factors_json)

    def count_by_kind(self, k: int) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT kind, COUNT(*) FROM factorizations WHERE k = ? GROUP BY kind",
            (k,),
        )
        return {kind: count for kind, count in cur}

    def close(self) -> None:
        self.conn.close()


class InMemoryRepository:
    """For tests."""
    def __init__(self) -> None:
        self._facts: dict[tuple[int, tuple[int, ...]], tuple[Factorization, Provenance]] = {}
        self._runs: list["RunRecord"] = []
        self._pending: list["PendingFactoring"] = []

    def add(self, fact: Factorization, provenance: Provenance) -> bool:
        key = (fact.k, fact.factors)
        if key in self._facts:
            return False
        self._facts[key] = (fact, provenance)
        return True

    def contains(self, fact: Factorization) -> bool:
        return (fact.k, fact.factors) in self._facts

    def all_lehmers(self, k: int) -> Iterator[Factorization]:
        for (kk, _), (f, _) in self._facts.items():
            if kk == k and f.is_lehmer():
                yield f

    def all_seeds(self, k: int) -> Iterator[Factorization]:
        for (kk, _), (f, _) in self._facts.items():
            if kk == k and f.is_plus_seed():
                yield f

    def by_length(self, k: int, length: int) -> Iterator[Factorization]:
        for (kk, _), (f, _) in self._facts.items():
            if kk == k and f.length == length:
                yield f

    def count_by_kind(self, k: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for (kk, _), (f, _) in self._facts.items():
            if kk == k:
                out[f.kind.value] = out.get(f.kind.value, 0) + 1
        return out

    def record_run(self, record: "RunRecord") -> int:
        record.id = len(self._runs) + 1
        self._runs.append(record)
        return int(record.id)

    def begin_run(self, strategy: str, k: int, max_r: int) -> int:
        from datetime import datetime, timezone
        from spoof_lehmer.tracking import RunRecord
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        record = RunRecord(
            id=None, strategy=strategy, k=k, max_r=max_r, max_N=None,
            node_limit=None, started_at=now, finished_at=now,
            status="interrupted", nodes_explored=0,
            factorizations_added=0, notes="in-progress",
        )
        return self.record_run(record)

    def mark_run_complete(self, run_id: int, record: "RunRecord") -> None:
        for r in self._runs:
            if r.id == run_id:
                r.finished_at = record.finished_at
                r.status = record.status
                r.nodes_explored = record.nodes_explored
                r.factorizations_added = record.factorizations_added
                r.notes = record.notes
                return

    def abandon_run(self, run_id: int) -> None:
        self._runs = [r for r in self._runs if r.id != run_id]

    def all_runs(self, k: int) -> list["RunRecord"]:
        return [r for r in self._runs if r.k == k]

    def add_pending(self, item: "PendingFactoring") -> bool:
        for existing in self._pending:
            if (existing.k == item.k and existing.seed_factors == item.seed_factors
                    and existing.delta_kind == item.delta_kind):
                return False
        self._pending.append(item)
        return True

    def all_pending(self, k: int) -> list["PendingFactoring"]:
        return [p for p in self._pending if p.k == k]

    def close(self) -> None:
        pass
