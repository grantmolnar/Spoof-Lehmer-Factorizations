"""Progress reporting for long-running enumerations.

The :class:`ProgressReporter` protocol exists so that the search
strategy and CLI can communicate without either depending on a
concrete logger implementation.  The strategy calls reporter hooks at
semantically meaningful moments (k started, length started, found a
factorization, heartbeat); the reporter decides what (if anything) to
print, how to format it, and when to throttle.

This separation is the Interface Segregation + Dependency Inversion
pieces of SOLID: the strategy depends on an abstract reporter, not on
stdout or a specific file.  Tests can pass a no-op reporter; the CLI
can pass a human-readable stderr reporter; future work (a TUI, a
file-based log, a structured JSONL stream) can plug in without
touching the strategy.

Two concrete reporters ship here:

  * :class:`SilentProgressReporter` --- no-op, used by tests and by
    callers that don't want any output.
  * :class:`StderrProgressReporter` --- human-readable progress to
    stderr with timestamped event lines and a throttled heartbeat
    for long inner loops.
"""
from __future__ import annotations

import sys
import time
from typing import Protocol, TextIO


class ProgressReporter(Protocol):
    """Protocol for progress-reporting during enumeration.

    Implementations receive events during :meth:`BoundsPropagationStrategy.discover`
    and :meth:`discover_subtree`.  All methods have default no-op
    implementations via :class:`SilentProgressReporter`; concrete
    reporters override whichever hooks they care about.
    """

    def on_k_start(self, k: int, max_r: int) -> None:
        """Called at the start of each k value's enumeration."""
        ...

    def on_k_end(
        self, k: int, max_r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        """Called at the end of each k value's enumeration."""
        ...

    def on_length_start(self, k: int, r: int) -> None:
        """Called at the start of each (k, r) search within a k loop."""
        ...

    def on_length_end(
        self, k: int, r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        """Called at the end of each (k, r) search."""
        ...

    def on_found(
        self, k: int, r: int, factors: tuple[int, ...], elapsed_since_k_start: float,
    ) -> None:
        """Called each time a new factorization is added to the repo."""
        ...

    def on_heartbeat(
        self, k: int, r: int, nodes_explored: int, elapsed_since_length_start: float,
    ) -> None:
        """Called periodically during long-running length searches."""
        ...


class SilentProgressReporter:
    """No-op reporter.  Default for non-interactive use (tests)."""

    def on_k_start(self, k: int, max_r: int) -> None:
        pass

    def on_k_end(
        self, k: int, max_r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        pass

    def on_length_start(self, k: int, r: int) -> None:
        pass

    def on_length_end(
        self, k: int, r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        pass

    def on_found(
        self, k: int, r: int, factors: tuple[int, ...], elapsed_since_k_start: float,
    ) -> None:
        pass

    def on_heartbeat(
        self, k: int, r: int, nodes_explored: int, elapsed_since_length_start: float,
    ) -> None:
        pass


class StderrProgressReporter:
    """Human-readable progress reporter writing timestamped lines to stderr.

    Emits:
      * a header line when each k value starts;
      * a one-line summary when each (k, r) length completes, including
        elapsed wall time;
      * an immediate line for every new factorization found (with
        factor list and cumulative k-loop wall time);
      * a throttled heartbeat during long length searches
        (configurable; default every 30 seconds).

    Attributes:
        stream: the file-like object to write to (default ``sys.stderr``).
        heartbeat_seconds: minimum interval between heartbeats during a
            single (k, r) length.  Set to 0 to disable heartbeats.
        found_factor_limit: if a factorization has more factors than
            this (rare for visual clarity), they are truncated in the
            on_found line.  Default 8 (the typical census max).
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        heartbeat_seconds: float = 30.0,
        found_factor_limit: int = 8,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.heartbeat_seconds = heartbeat_seconds
        self.found_factor_limit = found_factor_limit
        self._last_heartbeat: float = 0.0

    def _write(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.stream.write(f"[{ts}] {msg}\n")
        self.stream.flush()

    def on_k_start(self, k: int, max_r: int) -> None:
        self._write(f"k = {k}: starting enumeration (max_r = {max_r})")
        self._last_heartbeat = time.perf_counter()

    def on_k_end(
        self, k: int, max_r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        self._write(
            f"k = {k}: done  added={added}  nodes={nodes_explored:,}  "
            f"elapsed={elapsed:.1f}s"
        )

    def on_length_start(self, k: int, r: int) -> None:
        self._last_heartbeat = time.perf_counter()
        if r >= 7:
            # Only announce length starts for expensive r; shorter r is usually fast.
            self._write(f"  k = {k}, r = {r}: starting")

    def on_length_end(
        self, k: int, r: int, added: int, nodes_explored: int, elapsed: float,
    ) -> None:
        if r >= 5 or added > 0:
            self._write(
                f"  k = {k}, r = {r}: added={added}  "
                f"nodes={nodes_explored:,}  elapsed={elapsed:.1f}s"
            )

    def on_found(
        self, k: int, r: int, factors: tuple[int, ...], elapsed_since_k_start: float,
    ) -> None:
        if len(factors) <= self.found_factor_limit:
            disp = str(factors)
        else:
            head = ", ".join(str(f) for f in factors[:4])
            tail = ", ".join(str(f) for f in factors[-2:])
            disp = f"({head}, ..., {tail})  [length {len(factors)}]"
        self._write(
            f"    FOUND k = {k}, r = {r}: {disp}  (at {elapsed_since_k_start:.1f}s)"
        )

    def on_heartbeat(
        self, k: int, r: int, nodes_explored: int, elapsed_since_length_start: float,
    ) -> None:
        if self.heartbeat_seconds <= 0:
            return
        now = time.perf_counter()
        if now - self._last_heartbeat < self.heartbeat_seconds:
            return
        self._last_heartbeat = now
        self._write(
            f"    ... k = {k}, r = {r}: nodes={nodes_explored:,}  "
            f"elapsed={elapsed_since_length_start:.1f}s"
        )
