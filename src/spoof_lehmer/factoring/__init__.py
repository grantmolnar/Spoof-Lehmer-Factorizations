"""Factoring backends. Protocol + implementations.

Strategy: small n uses trial division, medium uses sympy, large uses yafu.
The AutoBackend dispatches to the right tool based on input size.
"""
from __future__ import annotations
from typing import Protocol


class FactoringBackend(Protocol):
    """Factor a positive integer into prime powers."""
    name: str
    max_n: int
    def factor(self, n: int) -> dict[int, int]: ...


class TrialDivisionBackend:
    """Trial division. Suitable for n up to ~10^12."""
    name = "trial"
    max_n = 10**12

    def factor(self, n: int) -> dict[int, int]:
        if n <= 1:
            return {}
        out: dict[int, int] = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                out[d] = out.get(d, 0) + 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            out[n] = out.get(n, 0) + 1
        return out


class SympyBackend:
    """sympy.factorint. Pollard rho + ECM. Suitable for n up to ~10^30."""
    name = "sympy"
    max_n = 10**42

    def factor(self, n: int) -> dict[int, int]:
        import sympy  # type: ignore[import-untyped]
        return dict(sympy.factorint(n))


class AutoBackend:
    """Dispatches to the cheapest backend that can handle the input.

    Order of preference (small to large):
        1. trial division (n <= 10^10, very fast for small n)
        2. sympy            (10^10 < n <= 10^30, no subprocess overhead)
        3. yafu             (10^30 < n, if installed)
    """
    name = "auto"

    def __init__(self) -> None:
        self._trial = TrialDivisionBackend()
        try:
            import sympy  # noqa: F401
            self._sympy: SympyBackend | None = SympyBackend()
        except ImportError:
            self._sympy = None
        try:
            from spoof_lehmer.factoring.yafu_backend import YafuBackend
            self._yafu: "YafuBackend | None" = YafuBackend()
        except (RuntimeError, ImportError):
            self._yafu = None

        self.max_n = (
            10**60 if self._yafu else (10**42 if self._sympy else 10**12)
        )

    def factor(self, n: int) -> dict[int, int]:
        if n <= 10**10:
            return self._trial.factor(n)
        if self._sympy is not None and n <= 10**30:
            return self._sympy.factor(n)
        if self._yafu is not None:
            return self._yafu.factor(n)
        if self._sympy is not None:
            return self._sympy.factor(n)
        return self._trial.factor(n)

    def describe(self) -> str:
        parts = ["trial"]
        if self._sympy:
            parts.append("sympy")
        if self._yafu:
            parts.append("yafu")
        return f"auto({'+'.join(parts)})"


def default_backend() -> FactoringBackend:
    """Returns AutoBackend, which picks the right tool per input."""
    return AutoBackend()
