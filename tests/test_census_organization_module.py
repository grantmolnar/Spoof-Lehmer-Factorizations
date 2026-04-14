"""Tests for spoof_lehmer.analysis.census_organization."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spoof_lehmer.analysis.census_organization import (
    ParentKind,
    classify,
    organize,
)

CENSUS_PATH = Path(__file__).resolve().parent.parent / "data" / "enumerate_odd_r7.json"


@pytest.fixture
def census_k2() -> list[tuple[int, ...]]:
    if not CENSUS_PATH.exists():
        pytest.skip(f"census not at {CENSUS_PATH}")
    raw = json.loads(CENSUS_PATH.read_text())
    return [tuple(d["factors"]) for d in raw if d["k"] == 2]


class TestClassify:
    def test_trivial_is_trivial(self) -> None:
        prov = classify((3, 3))
        assert prov.kind is ParentKind.TRIVIAL
        assert prov.parent is None

    def test_descended_factorization(self) -> None:
        # (3, 5, 17, 257, 65537, 4294967295) descends from (3, 5, 17, 257) via
        # x_5, x_6 = (65537, 4294967295) (Hasanalizade descent).
        prov = classify((3, 5, 17, 257, 65537, 4294967295))
        assert prov.kind is ParentKind.HASANALIZADE_DESCENDED
        assert prov.parent == (3, 5, 17, 257)

    def test_lehmer_companion(self) -> None:
        # (5, 5, 5, 43, 5375) is the Lehmer companion of plus-seed (5, 5, 5, 43)
        # since 5375 = eps((5, 5, 5, 43)).
        prov = classify((5, 5, 5, 43, 5375))
        assert prov.kind is ParentKind.LEHMER_COMPANION
        assert prov.parent == (5, 5, 5, 43)

    def test_truly_primitive(self) -> None:
        # (5, 5, 9, 9, 89): prefix (5, 5, 9) is not a plus-seed, prefix (5, 5, 9, 9)
        # is not a plus-seed.
        prov = classify((5, 5, 9, 9, 89))
        assert prov.kind is ParentKind.PRIMITIVE
        assert prov.parent is None


class TestOrganize:
    def test_total_count_matches_input(self, census_k2: list[tuple[int, ...]]) -> None:
        org = organize(census_k2)
        assert len(org.provenances) == len(census_k2)

    def test_partition_is_exhaustive(self, census_k2: list[tuple[int, ...]]) -> None:
        """Every input falls into exactly one of: descended, companion, primitive,
        trivial."""
        org = organize(census_k2)
        descended = sum(
            len(v) for v in org.descended_to_children.values()
        )
        companion = len(org.companion_to_child)
        primitive = sum(len(v) for v in org.primitives_by_length.values())
        trivial = sum(
            1 for p in org.provenances if p.kind is ParentKind.TRIVIAL
        )
        assert descended + companion + primitive + trivial == len(census_k2)

    def test_observed_decomposition(self, census_k2: list[tuple[int, ...]]) -> None:
        """Specific decomposition observed in the r<=7 census."""
        org = organize(census_k2)
        assert sum(1 for p in org.provenances if p.kind is ParentKind.TRIVIAL) == 1
        assert sum(
            1 for p in org.provenances if p.kind is ParentKind.HASANALIZADE_DESCENDED
        ) == 50
        assert sum(
            1 for p in org.provenances if p.kind is ParentKind.LEHMER_COMPANION
        ) == 11
        assert sum(
            1 for p in org.provenances if p.kind is ParentKind.PRIMITIVE
        ) == 41

    def test_chain_count(self, census_k2: list[tuple[int, ...]]) -> None:
        """12 chains observed: Fermat + 11 sporadic."""
        org = organize(census_k2)
        assert len(org.chains) == 12

    def test_chain_for_lookup(self, census_k2: list[tuple[int, ...]]) -> None:
        org = organize(census_k2)
        # (3, 5, 17, 257) is in the Fermat chain.
        chain = org.chain_for((3, 5, 17, 257))
        assert chain.is_fermat
        assert chain.root == ()

    def test_descendants_count_per_chain(
        self, census_k2: list[tuple[int, ...]]
    ) -> None:
        """The Fermat chain has 24 Lehmer descendants in r<=7 census."""
        org = organize(census_k2)
        fermat_chain = next(c for c in org.chains.values() if c.is_fermat)
        assert org.descendants_of_chain(fermat_chain) == 24

    def test_descended_children_sorted_by_x_minus_2(
        self, census_k2: list[tuple[int, ...]]
    ) -> None:
        """Each parent's descended children are sorted by x_{r-1}."""
        org = organize(census_k2)
        for parent, children in org.descended_to_children.items():
            x_minus_2 = [F[-2] for F in children]
            assert x_minus_2 == sorted(x_minus_2), (
                f"parent {parent}: children not sorted by x_{{r-1}}"
            )

    def test_companion_factor_equals_eps(
        self, census_k2: list[tuple[int, ...]]
    ) -> None:
        """For each Lehmer companion (P*, x), x = eps(P*)."""
        from math import prod
        org = organize(census_k2)
        for parent, child in org.companion_to_child.items():
            assert child[-1] == prod(parent), (
                f"companion {child}: last factor != eps(parent={parent})"
            )

    def test_unsupported_k_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            organize([(3, 3, 3)], k=3)


class TestSixFreshLength6Plus_Seeds:
    """Detection of fresh length-6 sporadic plus-seeds via inverse parent inference."""

    def test_six_fresh_chain_roots_at_length_6(
        self, census_k2: list[tuple[int, ...]]
    ) -> None:
        """The six fresh length-6 plus-seeds appear as chain roots."""
        org = organize(census_k2)
        length_6_roots = {c.root for c in org.chains.values() if len(c.root) == 6}
        expected = {
            (3, 5, 17, 257, 65729, 22318913),
            (3, 5, 17, 365, 855, 7234467),
            (3, 11, 11, 11, 677, 3659),
            (5, 5, 5, 43, 5413, 786349),
            (5, 5, 5, 53, 215, 158265),
            (5, 7, 7, 13, 13, 619),
        }
        assert length_6_roots == expected


class TestFreshPlusSeedsAtLength:
    def test_length_4(self, census_k2: list[tuple[int, ...]]) -> None:
        org = organize(census_k2)
        # Only one fresh plus-seed at length 4: the sporadic (5, 5, 5, 43).
        assert org.fresh_plus_seeds_at_length(4) == [(5, 5, 5, 43)]

    def test_length_5(self, census_k2: list[tuple[int, ...]]) -> None:
        org = organize(census_k2)
        # Four fresh plus-seeds at length 5.
        assert org.fresh_plus_seeds_at_length(5) == [
            (3, 5, 17, 353, 929),
            (3, 9, 9, 23, 131),
            (5, 5, 5, 47, 453),
            (5, 7, 7, 7, 133),
        ]

    def test_length_6(self, census_k2: list[tuple[int, ...]]) -> None:
        org = organize(census_k2)
        # Six fresh plus-seeds at length 6 (via inverse parent inference).
        assert org.fresh_plus_seeds_at_length(6) == [
            (3, 5, 17, 257, 65729, 22318913),
            (3, 5, 17, 365, 855, 7234467),
            (3, 11, 11, 11, 677, 3659),
            (5, 5, 5, 43, 5413, 786349),
            (5, 5, 5, 53, 215, 158265),
            (5, 7, 7, 13, 13, 619),
        ]

    def test_no_fresh_at_length_2_or_3(self, census_k2: list[tuple[int, ...]]) -> None:
        """The Fermat chain accounts for all plus-seeds at lengths 2, 3."""
        org = organize(census_k2)
        assert org.fresh_plus_seeds_at_length(2) == []
        assert org.fresh_plus_seeds_at_length(3) == []
