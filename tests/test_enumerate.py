"""Tests for the all-k enumeration behavior.

Reproduces the shape of Grant's original spoof-Lehmer-factorizations
repository: for a given (r, parity), find every k-Lehmer factorization
across the entire valid k-range.
"""
from spoof_lehmer.cli.enumerate import k_upper_bound
from spoof_lehmer.search import BoundsPropagationStrategy
from spoof_lehmer.storage import InMemoryRepository


def test_k_upper_bound_odd() -> None:
    # floor((3/2)^r) for small r
    assert k_upper_bound(2, is_even=False) == 2   # floor(9/4) = 2
    assert k_upper_bound(3, is_even=False) == 3   # floor(27/8) = 3
    assert k_upper_bound(4, is_even=False) == 5   # floor(81/16) = 5
    assert k_upper_bound(7, is_even=False) == 17  # floor(2187/128) = 17


def test_k_upper_bound_even() -> None:
    assert k_upper_bound(2, is_even=True) == 3      # 2^2 - 1
    assert k_upper_bound(3, is_even=True) == 7      # 2^3 - 1
    assert k_upper_bound(7, is_even=True) == 127    # 2^7 - 1


def test_enumerate_all_k_odd_r2() -> None:
    """Exhaustive all-k enumeration at r=2 odd. The only solution at r=2
    is (3,3) with k=2 (Lehmer ratio 8/4 = 2). No other k produces an
    odd-factor r=2 solution because k = (9-1)/4 = 2 is forced.
    """
    found: dict[int, set[tuple[int, ...]]] = {}
    for k in range(2, k_upper_bound(2, is_even=False) + 1):
        repo = InMemoryRepository()
        BoundsPropagationStrategy(
            k_target=k, max_r=2, is_even=False,
        ).discover(repo)
        found[k] = {f.factors for f in repo.all_lehmers(k)}
    assert found[2] == {(3, 3)}


def test_enumerate_all_k_even_r2_full_sweep() -> None:
    """At r=2 even, the only integer-k solutions are (3,3) with k=2
    (since 2*2*2 = 9-1 = 8) and (2,2) with k=3 (since 3*1*1 = 4-1 = 3).
    The all-k sweep must find both and nothing else.
    """
    found: dict[int, set[tuple[int, ...]]] = {}
    for k in range(2, k_upper_bound(2, is_even=True) + 1):
        repo = InMemoryRepository()
        BoundsPropagationStrategy(
            k_target=k, max_r=2, is_even=True,
        ).discover(repo)
        found[k] = {f.factors for f in repo.all_lehmers(k)}
    assert found == {2: {(3, 3)}, 3: {(2, 2)}}
