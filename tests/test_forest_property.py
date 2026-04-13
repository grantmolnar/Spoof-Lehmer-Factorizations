"""The forest property: every census in the database must satisfy
no Lehmer factorization has multiple descended pairs.

If this test ever fails, you've either found a counterexample to the
forest theorem (huge!) or a bug in the implementation. Either way,
you want to know immediately.
"""
from spoof_lehmer.analysis import run_census
from spoof_lehmer.domain import Factorization, find_descents
from spoof_lehmer.factoring import default_backend
from spoof_lehmer.search import RecurrenceStrategy, CascadeStrategy
from spoof_lehmer.storage import InMemoryRepository, Provenance


def test_forest_property_after_recurrence_k2() -> None:
    """After running the recurrence search for k=2, the result is a forest."""
    repo = InMemoryRepository()
    rec = RecurrenceStrategy(k=2, max_r=5, max_N=10**8, node_limit=1_000_000)
    rec.discover(repo)

    for L in repo.all_lehmers(2):
        descents = find_descents(L)
        assert len(descents) <= 1, (
            f"Forest property violated: {L.factors} has "
            f"{len(descents)} descents: {descents}"
        )


def test_forest_property_after_cascade_k2() -> None:
    """After bottom-up + cascade, still a forest."""
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=5, max_N=10**8, node_limit=1_000_000).discover(repo)
    CascadeStrategy(k=2, backend=default_backend(), max_rounds=2).discover(repo)

    for L in repo.all_lehmers(2):
        descents = find_descents(L)
        assert len(descents) <= 1


def test_forest_property_for_k3() -> None:
    """The forest theorem holds for all k. Spot-check k=3."""
    repo = InMemoryRepository()
    RecurrenceStrategy(k=3, max_r=5, max_N=10**8, node_limit=1_000_000).discover(repo)

    for L in repo.all_lehmers(3):
        descents = find_descents(L)
        assert len(descents) <= 1


def test_census_summary_k2() -> None:
    """Sanity: the small-bound census matches expected counts."""
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=5, max_N=10**8).discover(repo)

    report = run_census(repo, k=2)
    assert report.is_forest
    # Lehmer factorizations with r <= 5 and N <= 10^8:
    # (3,3), (3,5,15), (3,5,17,255), (3,5,17,285,2507),
    # (5,5,5,43,5375), (5,5,9,9,89). And r=5 (3,5,17,257,65535)
    # is at N ~ 4e9 so excluded.
    assert report.lehmers_total >= 5


def test_in_memory_repo_basic() -> None:
    repo = InMemoryRepository()
    f = Factorization((3, 3), k=2)
    assert repo.add(f, Provenance("test"))
    assert not repo.add(f, Provenance("test"))  # dedup
    assert repo.contains(f)
    assert list(repo.all_lehmers(2)) == [f]


def test_cascade_unbounded_terminates() -> None:
    """max_rounds=None must terminate naturally on small bounds."""
    repo = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=4, max_N=10**6).discover(repo)
    cascade = CascadeStrategy(k=2, backend=default_backend(), max_rounds=None)
    cascade.discover(repo)  # must not hang
    # After fixpoint, every Lehmer is still uniquely descended
    for L in repo.all_lehmers(2):
        assert len(find_descents(L)) <= 1


def test_cascade_unbounded_matches_high_cap() -> None:
    """Unbounded cascade gives same result as a generous explicit cap."""
    repo_a = InMemoryRepository()
    repo_b = InMemoryRepository()
    RecurrenceStrategy(k=2, max_r=4, max_N=10**6).discover(repo_a)
    RecurrenceStrategy(k=2, max_r=4, max_N=10**6).discover(repo_b)

    CascadeStrategy(k=2, backend=default_backend(), max_rounds=None).discover(repo_a)
    CascadeStrategy(k=2, backend=default_backend(), max_rounds=100).discover(repo_b)

    a_lehmers = {f.factors for f in repo_a.all_lehmers(2)}
    b_lehmers = {f.factors for f in repo_b.all_lehmers(2)}
    a_seeds = {f.factors for f in repo_a.all_seeds(2)}
    b_seeds = {f.factors for f in repo_b.all_seeds(2)}
    assert a_lehmers == b_lehmers
    assert a_seeds == b_seeds
