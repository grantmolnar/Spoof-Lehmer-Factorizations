from typing import Iterator, List, Optional, Tuple
from fractions import Fraction
from functools import reduce
import operator
from collections import Counter
from math import floor, ceil

# Let F = x1 ... xr be a spoof factorization all of whose exponents are 1
# We define F.evaluation = x1 * ... * xr, and F.totient = (x1 - 1) * ... * (xr - 1)
# We say F is nontrivial if xi != 0, 1 for all i. Necessarily, this also implies xi != -1.
# We say F is positive if xi > 0.
# We say F is a Lehmer factorization if F.evaluation - 1 = k * (F.totient) for some integer k. We define k(F) = (F.evaluation - 1)/(F.totient)
# F is clearly differentiable as long as all its arguments stay away from 1, and the derivative of F with respect to xi is
# (xi - F.evaluation)/(xi * (xi - 1) * F.totient) = 1/((xi - 1) * F.totient) - k(F)/(xi * (xi - 1))
# Suppose all the arguments of F are bounded away from the interval (0, 1).
# In this python file, we restrict our attention to positive factorizations

# This code assumes the following theorems, and no others:
# No interesting positive spoof Lehmer factorizations has 1 as a base
# All exponents of odd spoof Lehmer factorizations are 1
# k(F) is decreasing in its arguments
# No factor of a spoof Lehmer factorization can be of the form p x +/- 1, where p is another factor.


def compute_product_of_list(numbers: List[int]) -> int:
    """
    Calculates the product of all integers in a given list.

    Parameters:
        numbers (List[int]): A list of integers whose product is to be found.

    Returns:
        int: The product of all integers in the list. Returns 1 for an empty list.

    Example:
        >>> compute_product_of_list([1, 2, 3, 4])
        24
    """
    output = reduce(operator.mul, numbers, 1)
    assert isinstance(output, int), "We define our product only on integer inputs"
    return output


def evaluate(numbers) -> int:
    """
    Returns the product of the elements of our list. This is an alias for compute_product_of_list.

    Parameters:
        numbers (List[int]): A list of integers to be multiplied

    Returns:
        int: The product of the elements of our list

    Example:
        >>> evaluate([-2, -1, 1, 2, 3, -3])
        -36
    """
    return compute_product_of_list(numbers)


def compute_totient(numbers) -> List[int]:
    """
    Returns the product of the elements of our list, each with 1 subtracted

    Parameters:
        numbers (List[int]): A list of integers to be multiplied after subtracting 1

    Returns:
        int: The product of the elements of our list

    Example:
        >>> evaluate([-2, -1, 1, 2, 3, -3])
        0
    """
    return compute_product_of_list([factor - 1 for factor in numbers])


class partialPositiveSpoofLehmerFactorization:
    """
    This class represents a partial positive spoof Lehmer factorization. It encapsulates three key elements:
    an integer r, an optional k, and an optional list of factors.

    Attributes:
        r (int): The number of (positive) terms in our factorizations
        k (Optional[int]): An optional parameter that can adjust the factorization in some manner.
        factors (Optional[List[int]]): An optional list of integers that are considered factors in this factorization.
    """

    def __init__(
        self,
        r: int,
        k: Optional[int] = None,
        is_even: bool = False,
        factors: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes a new instance of the partialSpoofLehmerFactorization class.

        Parameters:
            r (int): The number of (positive) terms in our factorizations
            k (Optional[int]): An optional adjustment parameter, which we'd like to satisfy k * phi(F) = e(F) - 1
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.
        """
        self.k = k
        self.factors = sorted(factors) if factors is not None else []
        self.is_even = is_even
        # The number of positive and negative terms our completed factorization will have
        self.r = r
        # The number of terms our factorization has so far
        self.s = len(self.factors)
        # The number of factors we have should not exceed the number we *can* have
        assert (
            self.s <= self.r
        ), f"The number of positive factors given, {self.splus}, must be less than or equal to the number of positive factors required, {self.rplus}"

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.r != other.r:
            return False
        if self.k != other.k:
            return False
        if self.factors != other.factors:
            # This is stronger than checking splus and sminus
            return False
        return True

    def with_additional_factor(
        self, next_factor: int
    ):  # -> partialSpoofLehmerFactorization
        """
        Produces a new instance of spoofLehmerFactorization with an additional factor appended to the list of factors.

        Parameters:
            next_factor : int
                The new factor we include

        Returns:
            partialPositiveSpoofLehmerFactorization: A new instance of partialPositiveSpoofLehmerFactorization with the updated list of factors.
        """
        extended_factors = self.factors.copy()
        extended_factors.append(next_factor)
        # Create a new instance with the same rplus, rminus, and k values, and the extended_factors list
        return partialPositiveSpoofLehmerFactorization(
            r=self.r, k=self.k, is_even=self.is_even, factors=extended_factors
        )

        return new_instance

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the partialSpoofLehmerFactorization instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"partialSpoofLehmerFactorization(r={self.r}, k={self.k}, is_even={self.is_even}, factors={self.factors})"

    def totient(self) -> int:
        return compute_totient(self.factors)

    def evaluation(self) -> int:
        return evaluate(self.factors)

    def k_bounds(self) -> Tuple[Fraction, Fraction]:
        """
        Returns a lower and upper bound on how large the ratio k(F) = F.evaluation/F.totient can be for F a (nontrivial positive) spoof factorization extending self compatible with our rplus and rminus conditions, and under the sorting asserted by sort_by_magnitude_then_positivity.
        """
        # k(F) is decreasing in its arguments
        # so we let our negative terms be -|maximum magnitude of existing term| and our positive terms be infinity to obtain an lower bound
        # and we let our negative terms be -infinity and our positive terms be |maximum magnitude of an existing term| to obtain an upper bound
        new_term_count = self.r - self.s

        if len(self.factors) > 0:
            # Our terms are in ascending order, so the next term is no smaller than teh current one
            minimal_next_term = self.factors[-1]
        else:
            # If our list was empty before, the smallest it can be is 2
            if self.is_even:
                minimal_next_term = 2
            else:
                minimal_next_term = 3
        # Our lower bound is attained by letting all remaining factors be infinity.
        lower_bound = Fraction(
            evaluate(self.factors) - (1 if new_term_count == 0 else 0),
            compute_totient(self.factors),
        )
        # Our upper bound is attained by letting all remaining factors be our minimal_next_term
        positive_augmented_factors = self.factors + [minimal_next_term] * new_term_count
        # This is a lower bound if r- % 2 == 1, and an upper bound otherwise
        upper_bound = Fraction(
            evaluate(positive_augmented_factors) - 1,
            compute_totient(positive_augmented_factors),
        )
        return lower_bound, upper_bound


def yield_all_positive_spoof_Lehmer_factorizations_given_r_k_parity(
    r: int,
    k: int,
    base_spoof: Optional[partialPositiveSpoofLehmerFactorization] = None,
    is_even: bool = False,
    verbose: bool = True,
) -> Iterator[partialPositiveSpoofLehmerFactorization]:
    """
    Yields all interesting spoof Lehmer factorizations with given rplus, rminus, and k.

    Parameters:
            r (int): The number of terms in our factorization
            k (int): We'd like to satisfy k * phi(F) = e(F) - 1
            is_even (bool): False If True, we look at totally even spoofs. Otherwise, we look only at odd spoofs.
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.

    Yields:
        All nontrivial odd spoof Lehmer factorizations with rplus positive factors, rminus negative factors, and such that k = (evaluate(F) - 1)/(compute_totient(F))
    """

    def infinite_range(start: int, step: int = 1):
        # We yield all integers from n on up.
        n = start
        while True:
            yield n
            n += step

    assert r > 1, "Our methods only apply if we have at least two factors"
    # If our base spoof is none, we initialize an empty spoof
    if base_spoof == None:
        base_spoof = partialPositiveSpoofLehmerFactorization(r, k, is_even=is_even, factors=None)
    # If our base spoof is complete, we check if it works
    if base_spoof.r == base_spoof.s:
        if k * base_spoof.totient() == base_spoof.evaluation() - 1:
            yield base_spoof
    # Otherwise, our base_spoof is incomplete and we will augment it if we can
    else:
        # If our upper or lower bounds are incompatible with k, there is no need to go further.
        # If lower_bound >= k, we win (since we have already excluded the possibility that our factorization is complete)
        # If upper_b
        # if lower_bound <= k and upper_bound <= k:
        # We need to consider the case of augmenting with new positive factors, and of augmenting with new negative factors, separately
        if base_spoof.factors == []:
            if is_even:
                start_term = 2
            else:
                start_term = 3
        else:
            start_term = base_spoof.factors[-1]

        # We know that L(factors, next_factor) increases to shared_limit, and U(factors, next_factor) decreases to shared_limit.
        shared_limit = Fraction(base_spoof.evaluation(), base_spoof.totient())
        # We have three cases: shared_limit > k, shared_limit == k, or shared_limit < k
        # If shared_limit > k, then for next_factor sufficiently large we will have L(factors, next_factor) > k, and once this happens we can never revert
        # If shared_limit < k, then for next_factor sufficiently large we will have U(factors, next_factor) < k, and once this happens we can never revert
        # If shared_limit = k, then we attain it only as U's factors tend to infinity, which means that no finite choice of factors can attain that limit, so we can terminate our process
        # Thus if shared_limit = k, we can break

        if shared_limit != k:
            for next_factor in infinite_range(start_term, step=2):

                # We use the congruence condition from Theorem 2 of Lehmer's paper to discard as many cases as we can
                if all(next_factor % p != 1 for p in base_spoof.factors):
                    augmented_spoof = base_spoof.with_additional_factor(next_factor)
                    (
                        augmented_lower_bound,
                        augmented_upper_bound,
                    ) = augmented_spoof.k_bounds()
                    # This is bound to happen eventually
                    if augmented_upper_bound < k:
                        break
                    else:
                        # We only yield a spoof if L <= k 
                        if augmented_lower_bound <= k:
                            # We can be more refined in our handling of infinities
                            # print("Inequalities unsatisfied!")
                            for (
                                spoof
                            ) in yield_all_positive_spoof_Lehmer_factorizations_given_r_k_parity(
                                r, k, base_spoof=augmented_spoof, is_even = is_even, verbose=verbose
                            ):
                                yield spoof


def yield_all_positive_spoof_Lehmer_factorizations_given_r_parity(
    r: int,
    base_spoof: Optional[partialPositiveSpoofLehmerFactorization] = None,
    is_even: bool = False,
    verbose: bool = True,
) -> Iterator[partialPositiveSpoofLehmerFactorization]:
    """
        Yields all interesting positive spoof Lehmer factorizations with given r.


    Parameters:
            r (int): The number of terms in our factorization
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.

    Yields:
        All nontrivial odd spoof Lehmer factorizations with rplus positive factors and rminus negative factors.
    """
    assert r > 1, "Our methods only apply if we have at least two factors"
    # We establish the bounds within which we need to work
    if base_spoof == None:
        lower_bound, upper_bound = partialPositiveSpoofLehmerFactorization(
            r, None, is_even=is_even, factors=None
        ).k_bounds()
    else:
        lower_bound, upper_bound = base_spoof.k_bounds()
    lower_bound, upper_bound = ceil(lower_bound), floor(upper_bound)
    if verbose:
        print(f"k lies between {lower_bound} and {upper_bound}, inclusive")
    # For each of these bounds, we see what happens
    for k in range(lower_bound, upper_bound + 1):
        if verbose:
            print(f"k = {k}")
        # if not (k == 1): # and r == 2): # If k = 1 and r = 2, the solutions are all of the form n*(2 - n)
        for spoof in yield_all_positive_spoof_Lehmer_factorizations_given_r_k_parity(
            r, k, base_spoof=base_spoof, is_even = is_even, verbose=verbose
        ):
            yield spoof
