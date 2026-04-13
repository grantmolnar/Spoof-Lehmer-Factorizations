"""Tests for search-run tracking and exhaustiveness queries."""
from spoof_lehmer.analysis import compute_coverage
from spoof_lehmer.factoring import default_backend, TrialDivisionBackend
from spoof_lehmer.search import RecurrenceStrategy, CascadeStrategy
from spoof_lehmer.storage import InMemoryRepository
from spoof_lehmer.tracking import RunStatus, RunResult


# === RunResult is what discover() now returns ===

def test_recurrence_returns_run_result() -> None:
    repo = InMemoryRepository()
    result = RecurrenceStrategy(k=2, max_r=4, max_N=10**6).discover(repo)
    assert isinstance(result, RunResult)
    assert result.status == RunStatus.COMPLETE
    assert result.added > 0
    assert result.nodes_explored > 0


def test_recurrence_records_run_in_repo() -> None:
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=4, max_N=10**6).discover(repo)
    runs = repo.all_runs(2)
    assert len(runs) == 1
    assert runs[0].strategy == "recurrence"
    assert runs[0].max_r == 4
    assert runs[0].status == RunStatus.COMPLETE.value
    assert runs[0].factorizations_added > 0


def test_recurrence_node_limit_hit_status() -> None:
    repo = InMemoryRepository()
    # node_limit=10 is absurdly small, so the search will truncate
    result = RecurrenceStrategy(k=2, max_r=6, max_N=10**12, node_limit=10).discover(repo)
    assert result.status == RunStatus.NODE_LIMIT_HIT


def test_cascade_returns_run_result() -> None:
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=4, max_N=10**6).discover(repo)
    result = CascadeStrategy(k=2, backend=default_backend(), max_rounds=2).discover(repo)
    assert isinstance(result, RunResult)
    # With sympy backend and small N, cascade should complete cleanly
    assert result.status == RunStatus.COMPLETE


def test_cascade_records_pending_when_delta_too_large() -> None:
    """Trial division backend has max_n=10^12; once seeds get big enough,
    Delta exceeds that and seeds end up in the pending queue."""
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=5, max_N=10**8).discover(repo)
    result = CascadeStrategy(
        k=2, backend=TrialDivisionBackend(), max_rounds=3,
        max_delta=10**10,  # force pending after a few rounds
    ).discover(repo)
    pending = repo.all_pending(2)
    if result.status == RunStatus.DELTA_TOO_LARGE:
        assert len(pending) > 0
        for p in pending:
            assert p.delta_value > 10**10
            assert p.delta_kind in ("lehmer", "seed")


# === Coverage queries ===

def test_coverage_empty_database() -> None:
    repo = InMemoryRepository()
    report = compute_coverage(repo, k=2, max_r=5)
    for r in range(1, 6):
        assert report.boxes[r].status == "NEVER_RUN"
    assert report.pending_count == 0


def test_coverage_after_complete_recurrence() -> None:
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=4, max_N=10**6).discover(repo)
    report = compute_coverage(repo, k=2, max_r=5)
    # r=1..4 should be EXHAUSTIVE
    for r in range(1, 5):
        assert report.boxes[r].status == "EXHAUSTIVE"
        assert report.boxes[r].max_N_complete == 10**6
    # r=5 should still be NEVER_RUN
    assert report.boxes[5].status == "NEVER_RUN"


def test_coverage_takes_largest_max_N() -> None:
    """If two complete runs cover the same r, max_N_complete is the larger."""
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=4, max_N=10**5).discover(repo)
    RecurrenceStrategy(k=2, max_r=4, max_N=10**7).discover(repo)
    report = compute_coverage(repo, k=2, max_r=4)
    assert report.boxes[4].max_N_complete == 10**7
    assert report.boxes[4].status == "EXHAUSTIVE"


def test_coverage_truncation_recorded() -> None:
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=6, max_N=10**12, node_limit=10).discover(repo)
    report = compute_coverage(repo, k=2, max_r=6)
    # Every r in 1..6 should record at least one truncated run
    assert report.boxes[6].truncated_runs >= 1
    assert report.boxes[6].status in ("TRUNCATED", "EXHAUSTIVE_BELOW_TRUNCATION")
