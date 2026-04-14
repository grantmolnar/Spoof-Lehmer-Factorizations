"""Tests for the Fermat-prefix closedness theorem.

These tests verify both the algebraic claims (Fermat-s is a plus-seed,
gcd(M_s, F_i) = 1) and the bijection with divisor pairs against the
empirical r <= 7 census.
"""
from __future__ import annotations

from math import gcd, prod

from spoof_lehmer.analysis.fermat import (
    chain,
    chain_extend,
    count_fermat_completions,
    fermat_M,
    fermat_completions,
    fermat_prefix,
    is_plus_seed,
    verify_fermat_completion,
)


def test_fermat_prefix_is_plus_seed() -> None:
    """For each s in 1..6, F^{(s)} is a k=2 plus-seed: 2*phi - eps = 1."""
    for s in range(1, 7):
        F = fermat_prefix(s)
        eps = prod(F)
        phi = prod(f - 1 for f in F)
        assert 2 * phi - eps == 1, (
            f"s={s}: F={F} is not a plus-seed: 2*phi - eps = {2*phi - eps}"
        )


def test_M_s_coprime_to_prior_fermats() -> None:
    """Lemma: gcd(M_s, F_i) = 1 for all i < s.

    Proof relies on 2^{2^i} ≡ -1 mod F_i, hence 2^{2^j} ≡ 1 mod F_i
    for j > i. Then M_s = 2^{2^{s+1}} - 2^{2^s} - 1 ≡ 1 - 1 - 1 = -1
    mod F_i, so gcd is 1.
    """
    for s in range(1, 7):
        M = fermat_M(s)
        for i in range(s):
            F_i = 2 ** (2 ** i) + 1
            assert gcd(M, F_i) == 1, (
                f"s={s}, i={i}: gcd(M_s, F_{i}={F_i}) != 1"
            )
            # Stronger: M_s ≡ -1 mod F_i.
            assert M % F_i == F_i - 1, (
                f"s={s}, i={i}: M_s mod F_i should be -1 mod F_i"
            )


def test_fermat_completions_are_lehmer() -> None:
    """Every output of fermat_completions(s) is a valid k=2-Lehmer."""
    for s in range(1, 6):
        for fact in fermat_completions(s):
            assert verify_fermat_completion(fact, k=2), (
                f"s={s}: invalid Lehmer {fact}"
            )


def test_fermat_completions_are_sorted() -> None:
    """Each completion is sorted (non-decreasing)."""
    for s in range(1, 6):
        for fact in fermat_completions(s):
            assert list(fact) == sorted(fact), f"unsorted: {fact}"


def test_fermat_completion_count_matches_tau_over_2() -> None:
    """tau(M_s) / 2 = number of length-(s+2) extensions of F^{(s)}."""
    expected = {1: 1, 2: 1, 3: 2, 4: 4, 5: 16}
    for s, n in expected.items():
        assert count_fermat_completions(s) == n, (
            f"s={s}: count_fermat_completions = "
            f"{count_fermat_completions(s)}, expected {n}"
        )
        # And the iterator should yield exactly that many.
        actual = sum(1 for _ in fermat_completions(s))
        assert actual == n, f"s={s}: iterator yielded {actual}, expected {n}"


def test_fermat_completions_match_census_at_r_eq_s_plus_2() -> None:
    """For s such that s+2 <= 7, the completions should be exactly the
    Fermat-s subset of the empirical r<=7 census."""
    import json
    from pathlib import Path

    census_path = (
        Path(__file__).resolve().parent.parent / "data" / "enumerate_odd_r7.json"
    )
    if not census_path.exists():
        # Skip if census not present (e.g. fresh checkout without data).
        import pytest
        pytest.skip(f"census data not present at {census_path}")

    data = json.loads(census_path.read_text())
    for s in range(1, 6):
        r = s + 2
        F_s = fermat_prefix(s)
        # Census entries that are length-r and start with F_s.
        from_census = {
            tuple(d["factors"])
            for d in data
            if d["k"] == 2
            and d["r"] == r
            and tuple(d["factors"][: len(F_s)]) == F_s
        }
        from_theorem = set(fermat_completions(s))
        assert from_theorem == from_census, (
            f"s={s}, r={r}: theorem and census disagree.\n"
            f"  theorem only: {sorted(from_theorem - from_census)}\n"
            f"  census only:  {sorted(from_census - from_theorem)}"
        )


# Theorem 2: Fermat chain uniqueness.
# The unique length-(s+1) extension of F^{(s)} that is itself a
# k=2 plus-seed is F^{(s+1)} = (F^{(s)}, F_s).

def _is_plus_seed(prefix: tuple[int, ...], k: int = 2) -> bool:
    eps = prod(prefix)
    phi = prod(f - 1 for f in prefix)
    return k * phi - eps == 1


def test_chain_uniqueness_fermat_extends_to_fermat() -> None:
    """For each s in 1..6, the extension F^{(s)} + (F_s,) is a plus-seed.
    This is the existence half of Theorem 2.
    """
    for s in range(1, 7):
        F_s = 2 ** (2 ** s) + 1
        extended = fermat_prefix(s) + (F_s,)
        assert _is_plus_seed(extended, k=2), (
            f"F^{{({s+1})}} = {extended} should be a plus-seed but isn't"
        )


def test_chain_uniqueness_no_other_extension_is_plus_seed() -> None:
    """For each s in 1..5, no length-(s+1) extension of F^{(s)} other
    than F^{(s+1)} is a plus-seed. This is the uniqueness half of
    Theorem 2: x = E_s + 2 = F_s is the ONLY solution.

    Cap the search: we test a wide range of x around F_s to confirm
    no other works. The proof is algebraic so we don't need an
    exhaustive search — any reasonable sample is conclusive.
    """
    for s in range(1, 6):
        F_s = 2 ** (2 ** s) + 1
        prefix = fermat_prefix(s)
        # Test x in a range around F_s, both above and below.
        for x in range(max(prefix[-1], F_s - 100), F_s + 100, 2):
            if x == F_s:
                continue
            extended = prefix + (x,)
            assert not _is_plus_seed(extended, k=2), (
                f"Unexpected plus-seed: F^{{({s})}} + ({x},) = {extended}"
            )


# Theorem 3: cascade collapse. M_u at u=1 in Proposition 3 applied
# to F^{(s)} equals M_{s+1}.

def test_cascade_collapse_at_u_eq_1() -> None:
    """Proven algebraically in the paper. Verify numerically for
    s in 1..6.
    """
    for s in range(1, 7):
        E = 2 ** (2 ** s) - 1
        B = 2 ** (2 ** s)
        EB = E * B
        # C from Proposition 3 with A=1: C = E - 1 + 3*E^2 + 2*E^3.
        C = E - 1 + 3 * E ** 2 + 2 * E ** 3
        M_u_at_1 = 1 * C + 1 * EB + EB ** 2
        M_s_plus_1 = fermat_M(s + 1)
        assert M_u_at_1 == M_s_plus_1, (
            f"s={s}: M_u|_{{u=1}} = {M_u_at_1}, M_{{{s+1}}} = {M_s_plus_1}"
        )


# Tests for the general plus-seed chain extension theorem
# (Theorem of the trees paper §"Plus-seed chains").


def test_is_plus_seed_recognizes_known_plus_seeds() -> None:
    """Verify is_plus_seed on known examples."""
    # Known k=2 plus-seeds.
    seeds = [
        (3,),
        (3, 5),
        (3, 5, 17),
        (3, 5, 17, 257),
        (5, 5, 5, 43),
        (3, 5, 17, 257, 65537),
        (3, 5, 17, 353, 929),
        (3, 9, 9, 23, 131),
        (5, 5, 5, 43, 5377),
        (5, 5, 5, 47, 453),
        (5, 7, 7, 7, 133),
    ]
    for s in seeds:
        assert is_plus_seed(s, k=2), f"{s} should be a k=2 plus-seed"


def test_is_plus_seed_rejects_non_plus_seeds() -> None:
    """Verify is_plus_seed correctly rejects non-plus-seeds."""
    # k=2 Lehmer factorizations are NOT plus-seeds (k*phi = eps - 1, not + 1).
    non_seeds = [
        (3, 3),
        (3, 5, 15),
        (3, 5, 17, 255),
        (5, 5, 9, 9, 89),
    ]
    for s in non_seeds:
        assert not is_plus_seed(s, k=2), (
            f"{s} should NOT be a k=2 plus-seed"
        )


def test_chain_extend_produces_plus_seed() -> None:
    """For every plus-seed F^*, chain_extend(F^*) is also a plus-seed.

    This is the existence half of the chain extension theorem.
    """
    seeds = [
        (3,),
        (3, 5),
        (3, 5, 17),
        (3, 5, 17, 257),
        (5, 5, 5, 43),
        (5, 5, 5, 43, 5377),
    ]
    for s in seeds:
        ext = chain_extend(s)
        assert is_plus_seed(ext, k=2), (
            f"chain_extend({s}) = {ext} is not a plus-seed"
        )
        # The next factor should be E + 2.
        E = prod(s)
        assert ext[-1] == E + 2, (
            f"chain_extend({s}): expected next factor {E + 2}, got {ext[-1]}"
        )


def test_chain_extend_uniqueness_at_short_seeds() -> None:
    """Empirically test the uniqueness half: for short plus-seeds, no
    OTHER value of x makes (F^*, x) a plus-seed.

    Test by brute force on the four smallest plus-seeds: scan x in a
    range around the predicted E + 2 and confirm no other works.
    """
    seeds = [(3,), (3, 5), (3, 5, 17), (3, 5, 17, 257)]
    for s in seeds:
        E = prod(s)
        x_predicted = E + 2
        # Scan a wide range around the prediction.
        for x in range(max(s[-1], x_predicted - 100),
                       x_predicted + 101, 2):
            if x == x_predicted:
                continue
            extended = s + (x,)
            assert not is_plus_seed(extended, k=2), (
                f"Unexpected plus-seed: {s} + ({x},) = {extended}"
            )


def test_chain_first_few_steps_match_known() -> None:
    """The first few steps of the chain from (3) match the Fermat
    partial products.
    """
    fermat_chain = chain((3,), length=5)
    expected = [
        (3,),
        (3, 5),
        (3, 5, 17),
        (3, 5, 17, 257),
        (3, 5, 17, 257, 65537),
    ]
    assert fermat_chain == expected


def test_sporadic_chain_first_few_steps() -> None:
    """The chain from (5, 5, 5, 43) extends to known values."""
    sporadic = chain((5, 5, 5, 43), length=4)
    expected = [
        (5, 5, 5, 43),
        (5, 5, 5, 43, 5377),
        (5, 5, 5, 43, 5377, 28901377),
        # The fourth element: E of S^(6) = 5*5*5*43*5377*28901377.
        # = 28901375 * 28901377 = 28901375 * 28901377
        (5, 5, 5, 43, 5377, 28901377,
         5 * 5 * 5 * 43 * 5377 * 28901377 + 2),
    ]
    assert sporadic == expected
    # Verify the fourth entry is indeed a plus-seed.
    assert is_plus_seed(sporadic[-1], k=2)


# Tests for the generalized plus-seed closedness theorem
# (Theorem of the trees paper §"The first sporadic chain").

def test_coprimality_lemma_holds_for_all_known_plus_seeds() -> None:
    """For every plus-seed F^*, gcd(M, x_i) = 1 where M = E^2 + E - 1.

    Proof: x_i | E so M ≡ -1 mod x_i, hence gcd(M, x_i) = 1.
    """
    seeds = [
        (3,),
        (3, 5),
        (3, 5, 17),
        (3, 5, 17, 257),
        (5, 5, 5, 43),
        (3, 5, 17, 257, 65537),
        (3, 5, 17, 353, 929),
        (3, 9, 9, 23, 131),
        (5, 5, 5, 43, 5377),
        (5, 5, 5, 47, 453),
        (5, 7, 7, 7, 133),
    ]
    for s in seeds:
        E = prod(s)
        M = E * E + E - 1
        for x in s:
            assert gcd(M, x) == 1, (
                f"Coprimality fails for {s}: gcd(M={M}, x_i={x}) = {gcd(M, x)}"
            )


def test_plus_seed_closedness_count_matches_census() -> None:
    """For each plus-seed F^* of length s, the predicted count
    tau(M)/2 of length-(s+2) Lehmer extensions matches the census.
    """
    import json
    from pathlib import Path
    from sympy import divisor_count

    census_path = (
        Path(__file__).resolve().parent.parent / "data" / "enumerate_odd_r7.json"
    )
    if not census_path.exists():
        import pytest
        pytest.skip(f"census not present at {census_path}")

    data = json.loads(census_path.read_text())

    plus_seeds_by_length = {
        4: [(3, 5, 17, 257), (5, 5, 5, 43)],
        5: [
            (3, 5, 17, 257, 65537),
            (3, 5, 17, 353, 929),
            (3, 9, 9, 23, 131),
            (5, 5, 5, 43, 5377),
            (5, 5, 5, 47, 453),
            (5, 7, 7, 7, 133),
        ],
    }

    for length, seeds in plus_seeds_by_length.items():
        for ps in seeds:
            E = prod(ps)
            M = E * E + E - 1
            tau_over_2 = int(divisor_count(M)) // 2
            r_target = length + 2
            census_count = sum(
                1 for d in data
                if d["k"] == 2 and d["r"] == r_target
                and tuple(d["factors"][:length]) == ps
            )
            assert tau_over_2 == census_count, (
                f"plus-seed {ps}: predicted {tau_over_2} length-{r_target} "
                f"extensions, census has {census_count}"
            )


# Test for u_max bound (Proposition u_max for plus-seeds).

def test_umax_bound_for_plus_seeds() -> None:
    """For any k=2 plus-seed F* with eps = E, the s=r-3 identity at F*
    has u^3 - 3*EB*u - C > 0 at u = 2(E+1), confirming u_max <= 2E + 1.

    Algebraic identity: (2E+2)^3 - 3E(E+1)(2E+2) - (2E^3 + 3E^2 + E - 1)
                      = 9E^2 + 17E + 9.
    """
    for ps in [
        (3,), (3, 5), (3, 5, 17), (3, 5, 17, 257),
        (5, 5, 5, 43), (3, 5, 17, 257, 65537),
        (3, 5, 17, 353, 929), (5, 5, 5, 43, 5377),
    ]:
        E = prod(ps)
        B = E + 1
        C = 2 * E ** 3 + 3 * E ** 2 + E - 1
        u = 2 * B
        gap = u ** 3 - 3 * E * B * u - C
        expected = 9 * E ** 2 + 17 * E + 9
        assert gap == expected, (
            f"plus-seed {ps}, E={E}: identity check failed, gap={gap} != {expected}"
        )
        assert gap > 0, f"plus-seed {ps}: gap {gap} should be positive"
