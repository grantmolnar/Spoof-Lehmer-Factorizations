"""Tests for the chain-forest census organization.

These tests verify that the classification used in
scripts/emit_organized_census.py is consistent with the census data
and the chain-extension theorem.
"""
from __future__ import annotations

import json
from math import prod
from pathlib import Path

import pytest

CENSUS_PATH = Path(__file__).resolve().parent.parent / "data" / "enumerate_odd_r7.json"


def is_plus_seed(F: tuple[int, ...], k: int = 2) -> bool:
    return k * prod(x - 1 for x in F) == prod(F) + 1


def chain_parent(P: tuple[int, ...]) -> tuple[int, ...] | None:
    """If P = (Q, eps(Q) + 2) for plus-seed Q, return Q. Else None."""
    if len(P) == 0:
        return None
    Q = P[:-1]
    if not is_plus_seed(Q):
        return None
    E_Q = prod(Q) if Q else 1
    return Q if P[-1] == E_Q + 2 else None


@pytest.fixture
def census() -> list[dict]:
    if not CENSUS_PATH.exists():
        pytest.skip(f"census not at {CENSUS_PATH}")
    return json.loads(CENSUS_PATH.read_text())


def test_total_k2_count(census: list[dict]) -> None:
    """k=2 r<=7 has exactly 103 factorizations."""
    n = sum(1 for d in census if d["k"] == 2)
    assert n == 103


def test_classification_partition_is_complete(census: list[dict]) -> None:
    """Every k=2 entry falls into exactly one of: trivial, descended,
    companion, primitive.
    """
    counts = {"trivial": 0, "descended": 0, "companion": 0, "primitive": 0}
    for d in census:
        if d["k"] != 2:
            continue
        F = tuple(d["factors"])
        if len(F) < 3:
            counts["trivial"] += 1
        elif is_plus_seed(F[:-2]):
            counts["descended"] += 1
        elif is_plus_seed(F[:-1]):
            counts["companion"] += 1
        else:
            counts["primitive"] += 1
    assert sum(counts.values()) == 103
    # The exact decomposition observed in the census.
    assert counts == {
        "trivial": 1,
        "descended": 50,
        "companion": 11,
        "primitive": 41,
    }


def test_companion_factorizations_have_x_eq_eps(census: list[dict]) -> None:
    """For Lehmer-companion entries F = (P*, x), the last factor x
    should equal eps(P*).
    """
    for d in census:
        if d["k"] != 2:
            continue
        F = tuple(d["factors"])
        if len(F) < 3:
            continue
        if is_plus_seed(F[:-2]):
            continue  # Hasanalizade-descended; not a companion case.
        P_companion = F[:-1]
        if is_plus_seed(P_companion):
            assert F[-1] == prod(P_companion), (
                f"companion {F}: last factor {F[-1]} != eps(P*) = {prod(P_companion)}"
            )


def test_chain_inventory_count(census: list[dict]) -> None:
    """Exactly 12 chains observed: Fermat + 11 sporadic. Equivalently,
    the number of distinct chain roots among observed plus-seeds is 12.
    """
    seeds: set[tuple[int, ...]] = set()
    for d in census:
        if d["k"] != 2:
            continue
        F = tuple(d["factors"])
        if len(F) >= 2 and is_plus_seed(F[:-2]):
            seeds.add(F[:-2])
        if len(F) >= 1 and is_plus_seed(F[:-1]):
            seeds.add(F[:-1])

    # Add chain ancestors so each member is in `seeds`.
    closure = set()
    for P in list(seeds):
        cur = P
        while cur is not None and cur not in closure:
            closure.add(cur)
            cur = chain_parent(cur)
    seeds = closure

    def root(P: tuple[int, ...]) -> tuple[int, ...]:
        cur = P
        while True:
            par = chain_parent(cur)
            if par is None:
                return cur
            cur = par

    roots = {root(P) for P in seeds}
    assert len(roots) == 12, f"expected 12 chain roots, got {len(roots)}: {sorted(roots, key=len)}"


def test_six_fresh_length6_sporadic_chains(census: list[dict]) -> None:
    """The six fresh length-6 sporadic plus-seeds detected via parent
    inversion from r=7 Lehmer companions.
    """
    expected = {
        (3, 5, 17, 257, 65729, 22318913),
        (3, 5, 17, 365, 855, 7234467),
        (3, 11, 11, 11, 677, 3659),
        (5, 5, 5, 43, 5413, 786349),
        (5, 5, 5, 53, 215, 158265),
        (5, 7, 7, 13, 13, 619),
    }
    found: set[tuple[int, ...]] = set()
    for d in census:
        if d["k"] != 2 or len(d["factors"]) != 7:
            continue
        F = tuple(d["factors"])
        P = F[:-1]
        if len(P) != 6:
            continue
        if not is_plus_seed(P):
            continue
        # Is P fresh (its length-5 prefix not a plus-seed)?
        if not is_plus_seed(P[:-1]):
            found.add(P)
    assert found == expected, (
        f"missing: {expected - found}\nextra: {found - expected}"
    )
