"""The Factorization value object - the central immutable type."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from math import prod


class FactorizationKind(Enum):
    LEHMER = "lehmer"
    PLUS_SEED = "plus_seed"
    OTHER = "other"


@dataclass(frozen=True)
class Factorization:
    """A spoof factorization: a sorted tuple of integers >= 2.

    Immutable and hashable. Computes evaluation, totient, deficiency, and
    classification (Lehmer/plus-seed/other) lazily.
    """
    factors: tuple[int, ...]
    k: int = field(compare=False)

    def __post_init__(self) -> None:
        if any(x < 2 for x in self.factors):
            raise ValueError(f"Factors must be >= 2; got {self.factors}")
        if tuple(sorted(self.factors)) != self.factors:
            raise ValueError(f"Factors must be sorted; got {self.factors}")
        if self.k < 2:
            raise ValueError(f"k must be >= 2; got {self.k}")

    @cached_property
    def evaluation(self) -> int:
        return prod(self.factors) if self.factors else 1

    @cached_property
    def totient(self) -> int:
        return prod(x - 1 for x in self.factors) if self.factors else 1

    @cached_property
    def deficiency(self) -> int:
        return self.k * self.totient - self.evaluation

    @cached_property
    def kind(self) -> FactorizationKind:
        d = self.deficiency
        if d == -1 and len(self.factors) >= 2:
            return FactorizationKind.LEHMER
        if d == 1:
            return FactorizationKind.PLUS_SEED
        return FactorizationKind.OTHER

    def is_lehmer(self) -> bool:
        return self.kind == FactorizationKind.LEHMER

    def is_plus_seed(self) -> bool:
        return self.kind == FactorizationKind.PLUS_SEED

    @property
    def length(self) -> int:
        return len(self.factors)

    @classmethod
    def empty(cls, k: int) -> "Factorization":
        return cls(factors=(), k=k)

    def with_factors(self, *new: int) -> "Factorization":
        return Factorization(factors=tuple(sorted((*self.factors, *new))), k=self.k)

    def without(self, i: int, j: int) -> "Factorization":
        """Return F with factors at indices i, j removed."""
        rem = tuple(x for idx, x in enumerate(self.factors) if idx != i and idx != j)
        return Factorization(factors=rem, k=self.k)

    def __repr__(self) -> str:
        return f"F{self.factors}_k{self.k}"
