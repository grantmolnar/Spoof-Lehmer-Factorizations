"""Mathematical invariants and regression tests."""
from spoof_lehmer.domain import (
    Factorization, descent_holds, find_descents, is_primitive,
    lehmer_delta, seed_delta,
)


# === Sanity: known solutions ===

def test_known_lehmer_factorizations() -> None:
    """The 4 smallest 2-Lehmer factorizations from Molnar-Singh."""
    cases = [
        ((3, 3), 9),
        ((3, 5, 15), 225),
        ((3, 5, 17, 255), 65025),
        ((5, 5, 9, 9, 89), 180225),
    ]
    for factors, N in cases:
        f = Factorization(factors, k=2)
        assert f.evaluation == N
        assert f.is_lehmer(), f"{factors} should be 2-Lehmer"


def test_known_plus_seeds() -> None:
    """The Fermat chain plus the first sporadic seed."""
    seeds = [
        (3,), (3, 5), (3, 5, 17), (3, 5, 17, 257), (5, 5, 5, 43),
    ]
    for s in seeds:
        f = Factorization(s, k=2)
        assert f.is_plus_seed(), f"{s} should be a plus-seed"


def test_empty_is_plus_seed_for_k2() -> None:
    assert Factorization.empty(2).is_plus_seed()
    assert not Factorization.empty(3).is_plus_seed()


# === The descent formula ===

def test_descent_formula_no_k_argument() -> None:
    """The k-independence theorem, encoded as a type-level fact."""
    import inspect
    sig = inspect.signature(descent_holds)
    assert "k" not in sig.parameters
    assert set(sig.parameters) == {"N", "a", "b"}


def test_descent_formula_known_pairs() -> None:
    """Verify against pairs we computed by hand."""
    # (3,3) descends from empty: N=9, pair (3,3)
    assert descent_holds(N=9, a=3, b=3)
    # (3,5,15) descends via pair (5,15) from seed (3): N=225
    assert descent_holds(N=225, a=5, b=15)
    # (3,5,17,255): N=65025, pair (17,255)
    assert descent_holds(N=65025, a=17, b=255)


def test_find_descents_unique_for_known_lehmers() -> None:
    """Forest property on the smallest known cases."""
    for factors in [(3, 3), (3, 5, 15), (3, 5, 17, 255)]:
        L = Factorization(factors, k=2)
        descents = find_descents(L)
        assert len(descents) == 1


def test_known_primitive() -> None:
    """(5,5,9,9,89) is primitive."""
    f = Factorization((5, 5, 9, 9, 89), k=2)
    assert is_primitive(f)


# === Extension equations are k-independent ===

def test_lehmer_delta_no_k() -> None:
    assert lehmer_delta(15) == 15 * 15 + 15 - 1
    # Same for any "k" - the formula doesn't take one
    assert lehmer_delta(15) == 239


def test_seed_delta_no_k() -> None:
    assert seed_delta(15) == 15 * 15 + 15 + 1
    assert seed_delta(15) == 241


# === Validation ===

def test_factorization_rejects_unsorted() -> None:
    import pytest
    with pytest.raises(ValueError, match="sorted"):
        Factorization((5, 3), k=2)


def test_factorization_rejects_small() -> None:
    import pytest
    with pytest.raises(ValueError, match=">="):
        Factorization((1, 3), k=2)
