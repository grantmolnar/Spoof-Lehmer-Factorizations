"""Tests for the progress-reporter plumbing."""
from __future__ import annotations

from dataclasses import dataclass, field

from spoof_lehmer.search import (
    BoundsPropagationStrategy,
    SilentProgressReporter,
    StderrProgressReporter,
)
from spoof_lehmer.search.progress import ProgressReporter
from spoof_lehmer.storage import InMemoryRepository


@dataclass
class RecordingReporter:
    """Test double that records every event call."""
    k_starts: list[tuple[int, int]] = field(default_factory=list)
    k_ends: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    length_starts: list[tuple[int, int]] = field(default_factory=list)
    length_ends: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    founds: list[tuple[int, int, tuple[int, ...], float]] = field(default_factory=list)
    heartbeats: list[tuple[int, int, int, float]] = field(default_factory=list)

    def on_k_start(self, k: int, max_r: int) -> None:
        self.k_starts.append((k, max_r))

    def on_k_end(
        self, k: int, max_r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        self.k_ends.append((k, max_r, added, nodes_explored, elapsed))

    def on_length_start(self, k: int, r: int) -> None:
        self.length_starts.append((k, r))

    def on_length_end(
        self, k: int, r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        self.length_ends.append((k, r, added, nodes_explored, elapsed))

    def on_found(
        self, k: int, r: int, factors: tuple[int, ...], elapsed_since_k_start: float,
    ) -> None:
        self.founds.append((k, r, factors, elapsed_since_k_start))

    def on_heartbeat(
        self, k: int, r: int, nodes_explored: int, elapsed_since_length_start: float,
    ) -> None:
        self.heartbeats.append((k, r, nodes_explored, elapsed_since_length_start))


def test_silent_reporter_has_all_protocol_methods() -> None:
    """SilentProgressReporter should satisfy the ProgressReporter protocol."""
    rep: ProgressReporter = SilentProgressReporter()
    # All methods should be callable without raising.
    rep.on_k_start(2, 7)
    rep.on_k_end(2, 7, 0, 0, 0.0)
    rep.on_length_start(2, 5)
    rep.on_length_end(2, 5, 0, 0, 0.0)
    rep.on_found(2, 5, (3, 5, 17, 257, 65535), 0.0)
    rep.on_heartbeat(2, 5, 1000, 1.0)


def test_stderr_reporter_satisfies_protocol() -> None:
    """StderrProgressReporter should satisfy the ProgressReporter protocol."""
    import io
    sink = io.StringIO()
    rep: ProgressReporter = StderrProgressReporter(stream=sink)
    rep.on_k_start(2, 7)
    rep.on_k_end(2, 7, 1, 100, 0.1)
    assert "k = 2" in sink.getvalue()
    assert "done" in sink.getvalue()


def test_strategy_calls_k_hooks() -> None:
    """Strategy should call on_k_start and on_k_end during discover()."""
    rep = RecordingReporter()
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=2, max_r=3, is_even=False, progress=rep,
    )
    strat.discover(repo)
    assert rep.k_starts == [(2, 3)]
    assert len(rep.k_ends) == 1
    k, max_r, added, nodes, elapsed = rep.k_ends[0]
    assert (k, max_r) == (2, 3)
    assert added >= 1  # at least (3, 3) at r=2


def test_strategy_calls_length_hooks() -> None:
    """Strategy should call on_length_start/end for each r in [min_r, max_r]."""
    rep = RecordingReporter()
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=2, min_r=2, max_r=4, is_even=False, progress=rep,
    )
    strat.discover(repo)
    # Three lengths: r = 2, 3, 4.
    assert rep.length_starts == [(2, 2), (2, 3), (2, 4)]
    assert [(k, r) for k, r, *_ in rep.length_ends] == [(2, 2), (2, 3), (2, 4)]


def test_strategy_calls_on_found_for_each_new_factorization() -> None:
    """Every new factorization should produce an on_found event."""
    rep = RecordingReporter()
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=2, min_r=2, max_r=5, is_even=False, progress=rep,
    )
    strat.discover(repo)
    # Expected r <= 5 k=2 Lehmers:
    # r=2: (3,3); r=3: (3,5,15); r=4: (3,5,17,255); r=5: 4 of them.
    assert len(rep.founds) == 7
    factors_found = [f for k, r, f, elapsed in rep.founds]
    assert (3, 3) in factors_found
    assert (3, 5, 17, 257, 65535) in factors_found
    assert (5, 5, 5, 43, 5375) in factors_found


def test_strategy_does_not_emit_duplicate_finds() -> None:
    """If the same factorization is already in the repo, on_found should not fire."""
    rep = RecordingReporter()
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=2, min_r=2, max_r=3, is_even=False, progress=rep,
    )
    strat.discover(repo)
    initial_finds = len(rep.founds)
    # Re-run: every factorization already in the repo, nothing new.
    strat2 = BoundsPropagationStrategy(
        k_target=2, min_r=2, max_r=3, is_even=False, progress=rep,
    )
    strat2.discover(repo)
    assert len(rep.founds) == initial_finds


def test_strategy_calls_subtree_hooks() -> None:
    """discover_subtree should also call k_start/k_end and length hooks."""
    rep = RecordingReporter()
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=2, min_r=5, max_r=5, is_even=False, progress=rep,
    )
    # Use a valid length-4 prefix (Fermat-4) and look for length-5 extensions.
    result = strat.discover_subtree(repo, prefix=(3, 5, 17, 257))
    assert rep.k_starts == [(2, 5)]
    assert len(rep.k_ends) == 1
    assert rep.length_starts == [(2, 5)]
    assert [(k, r) for k, r, *_ in rep.length_ends] == [(2, 5)]
    assert result.added >= 1  # (3, 5, 17, 257, 65535) should be found


def test_default_reporter_is_silent() -> None:
    """When no progress reporter is passed, the strategy uses a silent one."""
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=2, max_r=3, is_even=False,  # no progress=
    )
    # Should complete without error.
    result = strat.discover(repo)
    assert result.added >= 1


def test_stderr_reporter_throttles_heartbeat() -> None:
    """Heartbeat should only fire after the throttle interval has passed."""
    import io
    sink = io.StringIO()
    rep = StderrProgressReporter(stream=sink, heartbeat_seconds=10.0)
    # Two rapid calls: only the initial one (or none) should render.
    rep.on_k_start(2, 7)  # resets _last_heartbeat
    sink.truncate(0)
    sink.seek(0)
    rep.on_heartbeat(2, 5, 100, 0.5)
    rep.on_heartbeat(2, 5, 200, 1.0)
    # With heartbeat_seconds=10 and elapsed <1s, neither heartbeat should print.
    assert sink.getvalue() == ""
