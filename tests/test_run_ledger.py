"""Tests for the interrupt-safe run-ledger flow on
InMemoryRepository. SQLiteRepository exercises the same methods in
the enumerate_all.py smoke tests.
"""
from spoof_lehmer.storage import InMemoryRepository
from spoof_lehmer.tracking import RunRecord, RunResult, RunStatus


def test_begin_run_creates_interrupted_row() -> None:
    repo = InMemoryRepository()
    run_id = repo.begin_run("bounds_propagation", k=2, max_r=5)
    assert run_id > 0
    runs = repo.all_runs(k=2)
    assert len(runs) == 1
    assert runs[0].status == "interrupted"
    assert runs[0].max_r == 5


def test_abandon_run_removes_row() -> None:
    repo = InMemoryRepository()
    run_id = repo.begin_run("bounds_propagation", k=2, max_r=5)
    repo.abandon_run(run_id)
    assert repo.all_runs(k=2) == []


def test_mark_run_complete_upgrades_status() -> None:
    repo = InMemoryRepository()
    run_id = repo.begin_run("bounds_propagation", k=2, max_r=5)
    record = RunRecord.started_now("bounds_propagation", k=2, max_r=5, max_N=None)
    record.finish(RunResult(added=3, status=RunStatus.COMPLETE, nodes_explored=42))
    repo.mark_run_complete(run_id, record)
    runs = repo.all_runs(k=2)
    assert len(runs) == 1
    assert runs[0].status == "complete"
    assert runs[0].factorizations_added == 3
    assert runs[0].nodes_explored == 42


def test_mark_run_complete_on_invalid_id_is_noop() -> None:
    repo = InMemoryRepository()
    record = RunRecord.started_now("bounds_propagation", k=2, max_r=5, max_N=None)
    record.finish(RunResult(added=0, status=RunStatus.COMPLETE, nodes_explored=0))
    repo.mark_run_complete(run_id=999, record=record)  # nonexistent
    assert repo.all_runs(k=2) == []


def test_elapsed_seconds_from_persisted_timestamps() -> None:
    """RunRecord.elapsed_seconds must be derivable after a
    mark_run_complete round-trip, so prior runs' timings survive
    process restarts.
    """
    import time
    repo = InMemoryRepository()
    run_id = repo.begin_run("bounds_propagation", k=2, max_r=5)
    time.sleep(0.05)
    record = RunRecord.started_now("bounds_propagation", k=2, max_r=5, max_N=None)
    # Simulate a non-trivial run by overriding started_at to match
    # what begin_run wrote, then finishing.
    stored = repo.all_runs(k=2)[0]
    record.started_at = stored.started_at
    record.finish(RunResult(added=0, status=RunStatus.COMPLETE, nodes_explored=1))
    repo.mark_run_complete(run_id, record)
    reloaded = repo.all_runs(k=2)[0]
    assert reloaded.elapsed_seconds >= 0.05
    assert reloaded.elapsed_seconds < 5.0  # sanity: not wildly off


def test_elapsed_seconds_handles_unparseable_timestamps() -> None:
    record = RunRecord(
        id=1, strategy="x", k=2, max_r=5, max_N=None, node_limit=None,
        started_at="not-a-date", finished_at="also-not",
        status="complete", nodes_explored=0, factorizations_added=0, notes="",
    )
    assert record.elapsed_seconds == 0.0
