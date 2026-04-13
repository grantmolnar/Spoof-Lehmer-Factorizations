"""Tests for primitives and sporadic seed analysis."""
from spoof_lehmer.analysis import (
    analyze_primitives,
    analyze_sporadic_seeds,
    classify_seed,
    is_fermat_seed,
)
from spoof_lehmer.analysis.sporadic import contains_fermat_seed
from spoof_lehmer.domain import Factorization
from spoof_lehmer.search import RecurrenceStrategy, CascadeStrategy
from spoof_lehmer.factoring import default_backend
from spoof_lehmer.storage import InMemoryRepository


def _populate(max_r: int = 5, max_N: int = 10**8) -> InMemoryRepository:
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=max_r, max_N=max_N).discover(repo)
    CascadeStrategy(k=2, backend=default_backend(), max_rounds=2).discover(repo)
    return repo


# === Sporadic classification ===

def test_fermat_seed_classification() -> None:
    fermat_seeds = [
        Factorization((), k=2),
        Factorization((3,), k=2),
        Factorization((3, 5), k=2),
        Factorization((3, 5, 17), k=2),
        Factorization((3, 5, 17, 257), k=2),
    ]
    for s in fermat_seeds:
        assert classify_seed(s) == "fermat", f"{s.factors} should be fermat"


def test_is_fermat_seed_strict() -> None:
    assert is_fermat_seed(Factorization((3, 5), k=2))
    assert not is_fermat_seed(Factorization((3, 5, 17, 257), k=3))  # wrong k


def test_fermat_derived_classification() -> None:
    # (3,5,17,353,929) contains the Fermat seed (3,5,17)
    f = Factorization((3, 5, 17, 353, 929), k=2)
    assert contains_fermat_seed(f)
    assert classify_seed(f) == "fermat-derived"


def test_sporadic_classification() -> None:
    sporadic = [
        Factorization((5, 5, 5, 43), k=2),
        Factorization((5, 7, 7, 7, 133), k=2),
    ]
    for s in sporadic:
        assert classify_seed(s) == "sporadic"


# === Primitives report ===

def test_primitives_report_finds_known() -> None:
    repo = _populate()
    report = analyze_primitives(repo, k=2)

    primitive_factors = {p.factors for p in report.primitives}
    assert (5, 5, 9, 9, 89) in primitive_factors
    assert (5, 5, 5, 43, 5375) in primitive_factors


def test_primitives_length_distribution() -> None:
    repo = _populate()
    report = analyze_primitives(repo, k=2)
    # All primitives in this range have length >= 5
    assert all(r >= 5 for r in report.length_distribution)
    assert sum(report.length_distribution.values()) == report.total


def test_primitives_residue_classes() -> None:
    repo = _populate()
    report = analyze_primitives(repo, k=2)
    # Residue mod 16 is computed for every modulus
    assert 16 in report.residue_distribution
    total_in_mod16 = sum(report.residue_distribution[16].values())
    assert total_in_mod16 == report.total


def test_primitives_almost_descents() -> None:
    repo = _populate()
    report = analyze_primitives(repo, k=2)
    # Every primitive should have at least one near-miss recorded
    for p in report.primitives:
        assert p in report.almost_descents
        # The closest near-miss should have a positive residual (else it
        # would be a real descent and the factorization wouldn't be primitive)
        assert report.almost_descents[p][0][2] > 0


# === Sporadic report ===

def test_sporadic_report_classifies_all() -> None:
    repo = _populate()
    report = analyze_sporadic_seeds(repo, k=2)
    total = sum(len(v) for v in report.classifications.values())
    assert total == report.total


def test_sporadic_report_finds_fermat_chain() -> None:
    repo = _populate()
    report = analyze_sporadic_seeds(repo, k=2)
    fermat_seeds = report.classifications.get("fermat", [])
    fermat_factors = {s.factors for s in fermat_seeds}
    # The empty seed and (3,) and (3,5) should all be classified as fermat
    assert () in fermat_factors
    assert (3,) in fermat_factors
    assert (3, 5) in fermat_factors


def test_sporadic_report_inclusion_graph() -> None:
    repo = _populate()
    report = analyze_sporadic_seeds(repo, k=2)
    # The empty seed should have at least one child (a 2-factor seed)
    empty = Factorization((), k=2)
    if empty in report.inclusion_graph:
        children = report.inclusion_graph[empty]
        assert all(c.length == 2 for c in children)
