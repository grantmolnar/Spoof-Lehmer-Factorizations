"""Property-based regression tests for BoundsPropagationStrategy.

These tests protect against subtle corruption of the enumeration
pipeline as we add new optimizations. The core property is:

    For every (k, r, parity) in a reasonable range, the discovered
    set is invariant under *which* optimizations are enabled.

Every search mode we support (plain recursion, finishing-feasibility,
future s=r-3 closed form, future pruning filters) must find EXACTLY
the same set of k-Lehmer factorizations. Anything else is a bug.
"""
from hypothesis import given, settings, strategies as st

from spoof_lehmer.search import BoundsPropagationStrategy
from spoof_lehmer.storage import InMemoryRepository


def _enumerate(
    k: int, max_r: int, is_even: bool, use_ff: bool,
) -> set[tuple[int, ...]]:
    """Run the strategy and collect every discovered k-Lehmer."""
    repo = InMemoryRepository()
    BoundsPropagationStrategy(
        k_target=k,
        max_r=max_r,
        is_even=is_even,
        use_finishing_feasibility=use_ff,
    ).discover(repo)
    return {f.factors for f in repo.all_lehmers(k)}


@given(
    k=st.integers(min_value=2, max_value=3),
    max_r=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=15, deadline=None)
def test_finishing_feasibility_invariance_odd(
    k: int, max_r: int,
) -> None:
    """Core correctness invariant for odd mode: toggling FF must
    not change the discovered set.
    """
    with_ff = _enumerate(k, max_r, is_even=False, use_ff=True)
    without_ff = _enumerate(k, max_r, is_even=False, use_ff=False)
    assert with_ff == without_ff, (
        f"FF toggle changed results at k={k}, max_r={max_r}, odd.\n"
        f"  only with FF: {sorted(with_ff - without_ff)}\n"
        f"  only without: {sorted(without_ff - with_ff)}"
    )


def test_finishing_feasibility_invariance_even_small() -> None:
    """Even mode at (k=3, r=2): plain recursion is bounded here.

    The (k=2, r=2, is_even=True) case is NOT tested against plain
    recursion because it hits a known degenerate behavior: from
    prefix (2,) at k=2, the upper bound U(a) = (2a-1)/(a-1) never
    drops below k=2, so the candidate loop runs to its 10^12 cap.
    This is a latent limitation of plain recursion that
    finishing-feasibility bypasses — not a regression we need to
    guard against via property testing.
    """
    with_ff = _enumerate(3, max_r=2, is_even=True, use_ff=True)
    without_ff = _enumerate(3, max_r=2, is_even=True, use_ff=False)
    assert with_ff == without_ff


@given(
    k=st.integers(min_value=2, max_value=5),
    max_r=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_odd_results_have_odd_factors_only(k: int, max_r: int) -> None:
    """Every factorization in odd mode must have all-odd factors.

    Catches bugs in the parity gating inside finish_one / finish_two
    or future finish_three.
    """
    results = _enumerate(k, max_r, is_even=False, use_ff=True)
    for factors in results:
        for x in factors:
            assert x % 2 == 1, (
                f"even factor {x} in odd-mode result {factors} "
                f"(k={k}, max_r={max_r})"
            )


@given(
    k=st.integers(min_value=2, max_value=5),
    max_r=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_results_are_sorted_factorizations_odd(
    k: int, max_r: int,
) -> None:
    """Every discovered factor tuple is non-decreasing (odd mode).

    Catches bugs where finish_one/two/three emit pairs in wrong order.
    """
    results = _enumerate(k, max_r, is_even=False, use_ff=True)
    for factors in results:
        assert list(factors) == sorted(factors), (
            f"unsorted factorization {factors}"
        )


@given(
    k=st.integers(min_value=2, max_value=5),
    max_r=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_results_satisfy_defining_equation_odd(
    k: int, max_r: int,
) -> None:
    """Every emitted factorization must satisfy k * phi*(F) = eps(F) - 1.

    This is the fundamental correctness check: the discovered tuple
    is a k-Lehmer factorization. Catches arithmetic bugs in
    finish_one/two that happen to keep the discovered set stable
    relative to plain recursion but are wrong in absolute terms.
    """
    results = _enumerate(k, max_r, is_even=False, use_ff=True)
    for factors in results:
        eps = 1
        phi = 1
        for x in factors:
            eps *= x
            phi *= (x - 1)
        assert k * phi == eps - 1, (
            f"defining equation violated: k={k}, factors={factors}, "
            f"k*phi={k*phi}, eps-1={eps-1}"
        )


@given(
    k=st.integers(min_value=2, max_value=5),
    max_r=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_results_respect_min_factor_odd(
    k: int, max_r: int,
) -> None:
    """All factors in odd mode must be >= 3."""
    results = _enumerate(k, max_r, is_even=False, use_ff=True)
    for factors in results:
        assert all(x >= 3 for x in factors), (
            f"factor below 3 in odd-mode {factors}"
        )


# Example-based regression tests: canonical factorizations the
# optimized path must always find.

def test_canonical_k2_odd_small() -> None:
    """Canonical k=2 odd factorizations at r <= 5 must always be found.
    This set is the forest paper's census restricted to small r.
    """
    results = _enumerate(k=2, max_r=5, is_even=False, use_ff=True)
    canonical = {
        (3, 3),
        (3, 5, 15),
        (3, 5, 17, 255),
        (3, 5, 17, 257, 65535),
        (3, 5, 17, 285, 2507),
        (5, 5, 5, 43, 5375),
        (5, 5, 9, 9, 89),
    }
    missing = canonical - results
    assert not missing, f"missing canonical: {missing}"


def test_canonical_k5_odd_r4() -> None:
    """(3,3,3,3) at k=5: since 5 * 2^4 = 80 = 3^4 - 1."""
    results = _enumerate(k=5, max_r=4, is_even=False, use_ff=True)
    assert (3, 3, 3, 3) in results


def test_canonical_k3_even_r2() -> None:
    """(2,2) at k=3: 3*1*1 = 3 = 4-1."""
    results = _enumerate(k=3, max_r=2, is_even=True, use_ff=True)
    assert (2, 2) in results


# Property tests for finish_three.
# These compare `use_finish_three=True` against the baseline
# `use_finishing_feasibility=True, use_finish_three=False`.
# Any divergence is a bug in finish_three since finishing-feasibility
# at r-1, r-2 is already our correctness oracle.


def _enumerate_with_ff3(
    k: int, max_r: int, is_even: bool,
) -> set[tuple[int, ...]]:
    repo = InMemoryRepository()
    BoundsPropagationStrategy(
        k_target=k, max_r=max_r, is_even=is_even,
        use_finishing_feasibility=True,
        use_finish_three=True,
    ).discover(repo)
    return {f.factors for f in repo.all_lehmers(k)}


@given(
    k=st.integers(min_value=2, max_value=5),
    max_r=st.integers(min_value=3, max_value=6),
)
@settings(max_examples=30, deadline=None)
def test_finish_three_invariance_odd(k: int, max_r: int) -> None:
    """finish_three must not change the discovered set when
    layered on top of finishing-feasibility."""
    baseline = _enumerate(k, max_r, is_even=False, use_ff=True)
    with_ff3 = _enumerate_with_ff3(k, max_r, is_even=False)
    assert baseline == with_ff3, (
        f"finish_three changed results at k={k}, max_r={max_r}, odd.\n"
        f"  only in baseline: {sorted(baseline - with_ff3)}\n"
        f"  only in ff3:      {sorted(with_ff3 - baseline)}"
    )


def test_finish_three_canonical_k2_odd_r5() -> None:
    """With finish_three on, the canonical r=5 k=2 odd census is intact."""
    results = _enumerate_with_ff3(k=2, max_r=5, is_even=False)
    expected = {
        (3, 3),
        (3, 5, 15),
        (3, 5, 17, 255),
        (3, 5, 17, 257, 65535),
        (3, 5, 17, 285, 2507),
        (5, 5, 5, 43, 5375),
        (5, 5, 9, 9, 89),
    }
    missing = expected - results
    assert not missing, f"missing canonical: {missing}"


def test_finish_three_canonical_r6_k2_odd() -> None:
    """The 22 canonical r=6 k=2 odd factorizations (from the live r=7
    run's completed r=6 phase) must all be found with finish_three on."""
    results = _enumerate_with_ff3(k=2, max_r=6, is_even=False)
    # A sample of the canonical r=6 finds from the live r=7 log:
    canonical = {
        (3, 5, 17, 257, 65537, 4294967295),
        (3, 5, 17, 257, 65555, 226112997),
        (3, 5, 17, 257, 65717, 23794275),
        (3, 5, 17, 257, 68975, 1314417),
        (3, 5, 17, 353, 929, 83623935),
        (3, 5, 17, 377, 1217, 2295),
        (3, 5, 17, 395, 1059, 2303),
        (3, 5, 33, 53, 69, 8343),
        (3, 9, 9, 23, 131, 732159),
        (5, 5, 5, 43, 5377, 28901375),
        (5, 5, 5, 45, 807, 267023),
        (5, 5, 5, 65, 129, 2325),
        (5, 7, 7, 7, 133, 228095),
    }
    # Restrict to length-6 factorizations for comparison
    found_r6 = {f for f in results if len(f) == 6}
    missing = canonical - found_r6
    assert not missing, f"missing canonical r=6 k=2: {missing}"


# Property tests for discover_subtree: partitioned enumeration must
# yield exactly the same set as the unpartitioned discover.

def _enumerate_via_subtrees(
    k: int, max_r: int, is_even: bool, prefix_length: int,
) -> set[tuple[int, ...]]:
    """Enumerate by partitioning over all valid length-`prefix_length`
    prefixes and unioning the results."""
    repo = InMemoryRepository()
    strat = BoundsPropagationStrategy(
        k_target=k, max_r=max_r, is_even=is_even,
    )
    # Generate every prefix of the requested length by exhaustive
    # iteration via discover_subtree(prefix=()).
    # Simpler: iterate candidate prefixes manually using the same
    # parity rules. For test stability, enumerate up to a coarse cap.
    min_first = 2 if is_even else 3
    step = 1 if is_even else 2

    def gen_prefixes(length: int, start: int) -> list[tuple[int, ...]]:
        if length == 0:
            return [()]
        out = []
        # Coarse cap; tests only need r <= 5.
        for x in range(start, 200, step):
            for rest in gen_prefixes(length - 1, x):
                out.append((x,) + rest)
        return out

    if prefix_length == 0:
        prefixes: list[tuple[int, ...]] = [()]
    else:
        prefixes = gen_prefixes(prefix_length, min_first)

    for prefix in prefixes:
        strat.discover_subtree(repo, prefix)

    return {f.factors for f in repo.all_lehmers(k)}


def test_subtree_partition_invariance_k2_r5_odd() -> None:
    """Enumerating r<=5 k=2 odd via length-2 prefix partition must
    match the unpartitioned discover()."""
    baseline = _enumerate(k=2, max_r=5, is_even=False, use_ff=True)
    partitioned = _enumerate_via_subtrees(
        k=2, max_r=5, is_even=False, prefix_length=2,
    )
    assert baseline == partitioned, (
        f"partitioned != baseline\n"
        f"  in baseline only: {sorted(baseline - partitioned)}\n"
        f"  in partition only: {sorted(partitioned - baseline)}"
    )


def test_subtree_partition_invariance_k2_r5_odd_len1() -> None:
    """Same property with length-1 prefix partition (just x_1)."""
    baseline = _enumerate(k=2, max_r=5, is_even=False, use_ff=True)
    partitioned = _enumerate_via_subtrees(
        k=2, max_r=5, is_even=False, prefix_length=1,
    )
    assert baseline == partitioned


def test_subtree_partition_invariance_k5_r4_odd() -> None:
    """Same property at k=5 (catches (3,3,3,3) which has all-equal factors)."""
    baseline = _enumerate(k=5, max_r=4, is_even=False, use_ff=True)
    partitioned = _enumerate_via_subtrees(
        k=5, max_r=4, is_even=False, prefix_length=2,
    )
    assert baseline == partitioned


# Property tests for use_residue_filter: toggling the cheap pre-
# factorization filter must not change the discovered set.

def _enumerate_with_filter(
    k: int, max_r: int, is_even: bool, use_filter: bool,
) -> set[tuple[int, ...]]:
    repo = InMemoryRepository()
    BoundsPropagationStrategy(
        k_target=k, max_r=max_r, is_even=is_even,
        use_finishing_feasibility=True,
        use_residue_filter=use_filter,
    ).discover(repo)
    return {f.factors for f in repo.all_lehmers(k)}


@given(
    k=st.integers(min_value=2, max_value=5),
    max_r=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_residue_filter_invariance_odd(k: int, max_r: int) -> None:
    """Toggling use_residue_filter must not change the discovered set."""
    with_filter = _enumerate_with_filter(k, max_r, is_even=False, use_filter=True)
    without_filter = _enumerate_with_filter(k, max_r, is_even=False, use_filter=False)
    assert with_filter == without_filter, (
        f"residue filter changed results at k={k}, max_r={max_r}, odd.\n"
        f"  only with filter: {sorted(with_filter - without_filter)}\n"
        f"  only without:     {sorted(without_filter - with_filter)}"
    )


def test_residue_filter_canonical_r6_k2_odd() -> None:
    """With the residue filter on, the full r=6 k=2 odd census of 22
    factorizations must be intact. This is the strongest example test
    because r=6 actually exercises finish_two heavily under k=2."""
    results = _enumerate_with_filter(k=2, max_r=6, is_even=False, use_filter=True)
    r6 = {f for f in results if len(f) == 6}
    assert len(r6) == 22, (
        f"expected 22 r=6 k=2 odd factorizations, got {len(r6)}"
    )
    # And without the filter:
    baseline = _enumerate_with_filter(k=2, max_r=6, is_even=False, use_filter=False)
    assert results == baseline
