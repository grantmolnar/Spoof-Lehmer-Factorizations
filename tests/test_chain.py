"""Tests for spoof_lehmer.analysis.chain."""
from __future__ import annotations

from math import prod

import pytest

from spoof_lehmer.analysis.chain import (
    Chain,
    chain_extend,
    chain_parent,
    chain_root,
    is_fresh_plus_seed,
    is_plus_seed,
)


class TestIsPlusSeed:
    def test_empty_tuple_is_plus_seed_at_k2(self) -> None:
        """Empty product: 2 * 1 = 1 + 1, so empty tuple is k=2 plus-seed."""
        assert is_plus_seed((), k=2)

    def test_empty_tuple_not_plus_seed_at_k_neq_2(self) -> None:
        assert not is_plus_seed((), k=3)
        assert not is_plus_seed((), k=4)

    def test_known_fermat_chain_members(self) -> None:
        for F in [(3,), (3, 5), (3, 5, 17), (3, 5, 17, 257)]:
            assert is_plus_seed(F, k=2)

    def test_sporadic_plus_seeds(self) -> None:
        for F in [(5, 5, 5, 43), (3, 5, 17, 353, 929), (3, 9, 9, 23, 131)]:
            assert is_plus_seed(F, k=2)

    def test_lehmer_factorizations_not_plus_seeds(self) -> None:
        """k=2 Lehmer factorizations satisfy k*phi = eps - 1, not eps + 1."""
        for F in [(3, 3), (3, 5, 15), (3, 5, 17, 255), (5, 5, 9, 9, 89)]:
            assert not is_plus_seed(F, k=2)


class TestChainExtend:
    def test_fermat_chain_step(self) -> None:
        assert chain_extend((3,)) == (3, 5)
        assert chain_extend((3, 5)) == (3, 5, 17)
        assert chain_extend((3, 5, 17, 257)) == (3, 5, 17, 257, 65537)

    def test_sporadic_chain_step(self) -> None:
        assert chain_extend((5, 5, 5, 43)) == (5, 5, 5, 43, 5377)

    def test_extension_is_plus_seed(self) -> None:
        for F in [(3,), (5, 5, 5, 43), (3, 5, 17, 257, 65537)]:
            ext = chain_extend(F)
            assert is_plus_seed(ext)
            assert ext[-1] == prod(F) + 2

    def test_raises_on_non_plus_seed(self) -> None:
        with pytest.raises(ValueError, match="not a k = 2 plus-seed"):
            chain_extend((3, 3))

    def test_raises_for_unsupported_k(self) -> None:
        with pytest.raises(NotImplementedError):
            chain_extend((3,), k=3)


class TestChainParent:
    def test_root_has_no_parent(self) -> None:
        assert chain_parent(()) is None
        # (3,) is the chain extension of () (since eps(()) = 1 and 1 + 2 = 3),
        # so chain_parent((3,)) == ().
        assert chain_parent((3,)) == ()
        # (5, 5, 5, 43) is fresh: not a chain extension of any plus-seed.
        assert chain_parent((5, 5, 5, 43)) is None

    def test_chain_step_recovers_parent(self) -> None:
        for parent in [(3,), (3, 5), (5, 5, 5, 43)]:
            child = chain_extend(parent)
            assert chain_parent(child) == parent

    def test_non_chain_extension_has_no_chain_parent(self) -> None:
        # (3, 5, 17, 257, 65555) extends F^4 = (3,5,17,257) but x=65555 != 65535+2=65537.
        # So it's not a chain step; chain_parent should be None.
        assert chain_parent((3, 5, 17, 257, 65555)) is None


class TestChainRoot:
    def test_root_is_self(self) -> None:
        assert chain_root((3,)) == ()  # Walks to empty Fermat root via (3,) -> ()? No.
        # Actually (3,) is itself a fresh root since chain_parent((3,)) is None.
        # The Fermat chain has root (), and (3,) = chain_extend(()) since empty
        # tuple has eps = 1, and 1 + 2 = 3. Let me check.
        assert chain_extend(()) == (3,)
        assert chain_parent((3,)) == ()
        assert chain_root((3,)) == ()
        assert chain_root((3, 5)) == ()
        assert chain_root((3, 5, 17, 257, 65537)) == ()

    def test_sporadic_root(self) -> None:
        assert chain_root((5, 5, 5, 43)) == (5, 5, 5, 43)
        assert chain_root((5, 5, 5, 43, 5377)) == (5, 5, 5, 43)


class TestIsFreshPlusSeed:
    def test_fermat_root_is_fresh(self) -> None:
        assert is_fresh_plus_seed(())

    def test_sporadic_root_is_fresh(self) -> None:
        assert is_fresh_plus_seed((5, 5, 5, 43))
        assert is_fresh_plus_seed((3, 5, 17, 353, 929))

    def test_chain_extension_not_fresh(self) -> None:
        assert not is_fresh_plus_seed((3,))  # = chain_extend(())
        assert not is_fresh_plus_seed((5, 5, 5, 43, 5377))

    def test_non_plus_seed_not_fresh(self) -> None:
        assert not is_fresh_plus_seed((3, 3))


class TestChain:
    def test_construction_requires_fresh_root(self) -> None:
        with pytest.raises(ValueError, match="not a fresh"):
            Chain(root=(3, 5))  # Not fresh: (3,) is its parent.

    def test_fermat_chain_label_default(self) -> None:
        chain = Chain(root=())
        assert chain.label == "Fermat"
        assert chain.is_fermat

    def test_sporadic_chain_label_default(self) -> None:
        chain = Chain(root=(5, 5, 5, 43))
        assert chain.label == "sporadic"
        assert not chain.is_fermat

    def test_members_up_to_length(self) -> None:
        fermat = Chain(root=())
        members = fermat.members(up_to_length=4)
        assert members == [(), (3,), (3, 5), (3, 5, 17), (3, 5, 17, 257)]

    def test_member_at_depth(self) -> None:
        fermat = Chain(root=())
        assert fermat.member_at_depth(0) == ()
        assert fermat.member_at_depth(1) == (3,)
        assert fermat.member_at_depth(4) == (3, 5, 17, 257)

    def test_depth_of(self) -> None:
        fermat = Chain(root=())
        assert fermat.depth_of(()) == 0
        assert fermat.depth_of((3, 5, 17, 257)) == 4

    def test_depth_of_raises_for_non_member(self) -> None:
        fermat = Chain(root=())
        with pytest.raises(ValueError, match="not in chain"):
            fermat.depth_of((5, 5, 5, 43))

    def test_iter_members_starts_at_root(self) -> None:
        fermat = Chain(root=())
        it = fermat.iter_members()
        assert next(it) == ()
        assert next(it) == (3,)
        assert next(it) == (3, 5)
