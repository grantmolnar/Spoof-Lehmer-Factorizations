"""Pure mathematical core. No I/O dependencies."""
from spoof_lehmer.domain.factorization import Factorization, FactorizationKind
from spoof_lehmer.domain.descent import DescentPair, descent_holds, find_descents, is_primitive
from spoof_lehmer.domain.extension import (
    FactorFn,
    lehmer_delta,
    seed_delta,
    extensions_from_seed,
    divisor_pairs_from_factorization,
)

__all__ = [
    "Factorization", "FactorizationKind",
    "DescentPair", "descent_holds", "find_descents", "is_primitive",
    "FactorFn", "lehmer_delta", "seed_delta",
    "extensions_from_seed", "divisor_pairs_from_factorization",
]
