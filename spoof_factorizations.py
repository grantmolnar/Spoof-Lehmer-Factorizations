from typing import Iterator, List, Optional, Tuple
from fractions import Fraction
from functools import reduce
import operator
from collections import Counter
from math import floor, ceil
# Let F = x1 ... xr be a spoof factorization all of whose exponents are 1
# We define F.evaluation = x1 * ... * xr, and F.totient = (x1 - 1) * ... * (xr - 1)
# We say F is nontrivial if xi != 0, 1 for all i. Necessarily, this also implies xi != -1.
# We say F is odd if xi is odd for all i.
# We say F is a Lehmer factorization if F.evaluation - 1 = k * (F.totient) for some integer k. We define k(F) = (F.evaluation - 1)/(F.totient)
# F is clearly differentiable as long as all its arguments stay away from 1, and the derivative of F with respect to xi is
# (xi - F.evaluation)/(xi * (xi - 1) * F.totient) = 1/((xi - 1) * F.totient) - k(F)/(xi * (xi - 1))
# Suppose all the arguments of F are bounded away from the interval (0, 1).
# As a consequence, the signs of the factors of F are the same as the signs of the factors of F.totient, and so sign(F.evaluation) = sign(F.totient). (Thus k(F) is positive.)
# Let r- be the number of negative factors xi, and r+ be the number of positive factors, so r- + r+ = r.
# If r- is odd, then F.evaluation < 0, so the numerator of our equation is negative. Also, xi * (xi - 1) is positive regardless of whether xi is positive or negative, and F.totient is negative,
# So in this case, k(F) is increasing in its arguments.
# If r+ is even, then F.evaluation > 0, our numerator is positive, and a similar argument shows that k(F) is decreasing in its arguments.

# This code assumes the following theorems, and no others:
# No interesting odd spoof Lehmer factorizations have +/- 1 as a base
# All exponents of odd spoof Lehmer factorizations are 1
# k(F) is decreasing in its arguments if rminus is even, and increasing otherwise (proven above.)

# Notably, we do not (yet) use any congruence conditions on the factors


def count_positives_negatives(numbers: List[int]) -> Tuple[int, int]:
    """
    Counts the positive and negative integers in a given list.

    This function uses the Counter class from the collections module to efficiently
    categorize and count the numbers based on their sign. Zeroes are not counted
    as either positive or negative.

    Parameters:
        numbers (List[int]): A list of integers where positive, negative, and optionally zero
                              values can be present.

    Returns:
        Tuple[int, int]: A tuple where the first element is the count of positive integers,
                         and the second element is the count of negative integers in the list.
    """
    # We confirm every element of numbers is a real number
    assert all(
        isinstance(elem, (int, float, Fraction)) for elem in numbers
    ), "Our list of numbers should consist solely of numbers"
    # Create a Counter object by categorizing each number in the list based on its sign.
    # This approach iterates through 'numbers', applying a conditional expression to each
    # element to categorize it, and then counts the occurrences of each category.
    counter = Counter(
        "positive" if num > 0 else "negative" if num < 0 else "zero" for num in numbers
    )

    # The Counter object, 'counter', now has keys for 'positive', 'negative', and 'zero'
    # with their respective counts. We retrieve the counts for 'positive' and 'negative'
    # categories. If there are no positive or negative numbers, default to 0 count.
    positive_count = counter["positive"]
    negative_count = counter["negative"]

    return positive_count, negative_count


def sort_by_magnitude_then_positivity(numbers: List[int]) -> List[int]:
    """
    Sorts a list of integers by their magnitude from least to greatest. In case of a tie,
    positive numbers will come before negative ones.

    Parameters:
        numbers (List[int]): A list of integers to be sorted.

    Returns:
        List[int]: The sorted list of integers, with positive numbers coming before negative
                   numbers in case of a magnitude tie.

    Example:
        >>> sort_by_magnitude_then_positivity([-2, -1, 1, 2, 3, -3])
        [1, -1, 2, -2, 3, -3]
    """
    # Sort primarily by the absolute value, and in case of ties (same magnitude),
    # positive numbers should come before negative ones. This is achieved by
    # checking if the number is negative to influence the secondary sort criteria.
    return sorted(numbers, key=lambda x: (abs(x), 1 if x < 0 else 0))


def integer_magnitude_iterator(start: int, step: int = 1) -> Iterator[int]:
    """
    An iterator that yields integers with a magnitude greater than or equal to
    the given integer, starting with positives followed by negatives.
    If the given integer is negative, a positive integer of its own magnitude
    is not yielded.

    Parameters:
        start (int): The integer from which to start.
        step (int): The increment by which we vary our start.

    Yields:
        int: The next integer in the sequence.
    """
    assert step > 0, "Step size must be greater than 0"
    n = start
    if n <= 0:
        yield n
        n = abs(n) + step
    while True:
        yield n
        yield -n
        n += step


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


class partialSpoofLehmerFactorization:
    """
    This class represents a partial spoof Lehmer factorization. It encapsulates four key elements:
    rplus, rminus, an optional k, and an optional list of factors.

    Attributes:
        rplus (int): Represents the positive component of the factorization.
        rminus (int): Represents the negative component of the factorization.
        k (Optional[int]): An optional parameter that can adjust the factorization in some manner.
        factors (Optional[List[int]]): An optional list of integers that are considered factors in this factorization.
    """

    def __init__(
        self,
        rplus: int,
        rminus: int,
        k: Optional[int] = None,
        factors: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes a new instance of the partialSpoofLehmerFactorization class.

        Parameters:
            rplus (int): The positive component of the factorization.
            rminus (int): The negative component of the factorization.
            k (Optional[int]): An optional adjustment parameter, which we'd like to satisfy k * phi(F) = e(F) - 1
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.
        """
        self.k = k
        self.factors = (
            sort_by_magnitude_then_positivity(factors) if factors is not None else []
        )
        # The number of positive and negative terms our completed factorization will have
        self.rplus, self.rminus = rplus, rminus
        # The number of positive and negative terms our factorization has so far
        self.splus, self.sminus = count_positives_negatives(self.factors)
        # The number of positive and negative factors we have should not exceed the number we *can* have
        assert (
            self.splus <= self.rplus
        ), f"The number of positive factors given, {self.splus}, must be less than or equal to the number of positive factors required, {self.rplus}"
        assert (
            self.sminus <= self.rminus
        ), f"The number of negative factors given, {self.sminus}, must be less than or equal to the number of positive factors required, {self.rminus}"
        # self.evaluation = compute_product_of_list(self.factors)
        # self.totient = compute_product_of_list([factor - 1 for factor in self.factors])

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.rplus != other.rplus:
            return False
        if self.rminus != other.rminus:
            return False
        if self.k != other.k:
            return False
        if self.factors != other.factors:
            # This is stronger than checking splus and sminus
            return False
        return True

    def with_additional_factor(self, next_factor : int):  # -> partialSpoofLehmerFactorization
        """
        Produces a new instance of spoofLehmerFactorization with additional factors appended to the list of factors.

        Parameters:
            kwargs are new factors we can include

        Returns:
            spoofLehmerFactorization: A new instance of spoofLehmerFactorization with the updated list of factors.
        """
        extended_factors = self.factors.copy()
        extended_factors.append(next_factor)
        # Create a new instance with the same rplus, rminus, and k values, and the extended_factors list
        return partialSpoofLehmerFactorization(
            rplus=self.rplus, rminus=self.rminus, k=self.k, factors=extended_factors
        )

        return new_instance

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the partialSpoofLehmerFactorization instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"partialSpoofLehmerFactorization(rplus={self.rplus}, rminus={self.rminus}, k={self.k}, factors={self.factors})"

    def totient(self) -> int:
        return compute_totient(self.factors)

    def evaluation(self) -> int:
        return evaluate(self.factors)

    def k_bounds(self) -> Tuple[Fraction, Fraction]:
        """
        Returns an upper and lower bound on how large the ratio k(F) = F.evaluation/F.totient can be for F a (nontrivial odd) spoof factorization extending self compatible with our rplus and rminus conditions, and under the sorting asserted by sort_by_magnitude_then_positivity.
        """
        # k(F) is decreasing in its arguments
        # so we let our negative terms be -|maximum magnitude of existing term| and our positive terms be infinity to obtain an lower bound
        # and we let our negative terms be -infinity and our positive terms be |maximum magnitude of an existing term| to obtain an upper bound
        new_positive_term_count = self.rplus - self.splus
        # The number of new negative terms we will need to augment by
        new_negative_term_count = self.rminus - self.sminus

        if len(self.factors) > 0:
            # Our negative term is the smallest negative it can be
            negative_term = -abs(self.factors[-1])
            positive_term = abs(self.factors[-1])
        else:
            # If our list was empty before, the smallest it can be is -3
            negative_term = -3
            positive_term = 3
        negative_augmented_factors = (
            self.factors + [negative_term] * new_negative_term_count
        )
        # This is an upper bound if r- % 2 == 1, and a lower bound otherwise
        lower_bound = Fraction(
            evaluate(negative_augmented_factors)
            - (1 if new_positive_term_count == 0 else 0),
            compute_totient(negative_augmented_factors),
        )
        positive_augmented_factors = (
            self.factors + [positive_term] * new_positive_term_count
        )
        # This is a lower bound if r- % 2 == 1, and an upper bound otherwise
        upper_bound = Fraction(
            evaluate(positive_augmented_factors)
            - (1 if new_negative_term_count == 0 else 0),
            compute_totient(positive_augmented_factors),
        )
        return lower_bound, upper_bound


def yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k(
    rplus: int,
    rminus: int,
    k: int,
    base_spoof: Optional[partialSpoofLehmerFactorization] = None,
    verbose : bool = True
) -> Iterator[partialSpoofLehmerFactorization]:
    """
    Yields all interesting spoof Lehmer factorizations with given rplus, rminus, and k.

    Parameters:
            rplus (int): The positive component of the factorization.
            rminus (int): The negative component of the factorization.
            k (int): An optional adjustment parameter, which we'd like to satisfy k * phi(F) = e(F) - 1
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.

    Yields:
        All nontrivial odd spoof Lehmer factorizations with rplus positive factors, rminus negative factors, and such that k = (evaluate(F) - 1)/(compute_totient(F))
    """
    assert rplus + rminus > 1, "Our methods only apply if we have at least two factors"
    # If our base spoof is none, we initialize an empty spoof
    if base_spoof == None:
        base_spoof = partialSpoofLehmerFactorization(rplus, rminus, k, factors=None)
    # If our base spoof is complete, we check if it works

    if (
        base_spoof.rplus == base_spoof.splus and base_spoof.rminus == base_spoof.sminus
    ):
        # print(k, base_spoof.totient(), base_spoof.evaluation() - 1)
        if k * base_spoof.totient() == base_spoof.evaluation() - 1:
            yield base_spoof
    # Otherwise, our base_spoof is incomplete and we will augment it if we can
    else:
        # If our upper or lower bounds are incompatible with k, there is no need to go further.
        # if lower_bound <= k and upper_bound <= k:
        # We need to consider the case of augmenting with new positive factors, and of augmenting with new negative factors, separately
        if base_spoof.factors == []:
            start_term = 3
        else:
            start_term = base_spoof.factors[-1]
        # Even if the current case is impossible, we need to double-check the next case, too, because positive and negative terms are different. After that, we're good
        fail_on_positive = False
        fail_on_negative = False
        # We step by 2 from our start term because we know we want odd factors
        for next_factor in integer_magnitude_iterator(start_term, step=2):
            # print(next_factor)
            if next_factor > 0 and base_spoof.rplus == base_spoof.splus:
                fail_on_positive = True
                # print("Failed on positive")
            elif next_factor < 0 and base_spoof.rminus == base_spoof.sminus:
                fail_on_negative = True
                # print("Failed on negative")
            # We use the congruence condition from Theorem 2 of Lehmer's paper to discard as man cases as we can
            elif all(next_factor % p != 1 for p in base_spoof.factors):
                augmented_spoof = base_spoof.with_additional_factor(next_factor)
                augmented_lower_bound, augmented_upper_bound = (
                    augmented_spoof.k_bounds()
                )
                # if verbose:
                #    print(augmented_lower_bound, k, augmented_upper_bound)
                # print(k > augmented_upper_bound)
                if k < augmented_lower_bound or k > augmented_upper_bound:
                    # print("Inequalities satisfied!")
                    if next_factor > 0:
                        fail_on_positive = True
                        # print("Failed on positive")
                    else:
                        fail_on_negative = True
                        # print("Failed on negative")
                else:
                    # print("Inequalities unsatisfied!")
                    for spoof in yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k(
                            rplus, rminus, k, base_spoof=augmented_spoof, verbose = verbose
                        ):
                        yield spoof
                if fail_on_positive and fail_on_negative:
                    break


def yield_all_spoof_Lehmer_factorizations_given_rplus_rminus(
    rplus: int,
    rminus: int,
    base_spoof: Optional[partialSpoofLehmerFactorization] = None,
    verbose: bool = True,
) -> Iterator[partialSpoofLehmerFactorization]:
    """
        Yields all interesting spoof Lehmer factorizations with given rplus, rminus.


    Parameters:
            rplus (int): The positive component of the factorization.
            rminus (int): The negative component of the factorization.
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.

    Yields:
        All nontrivial odd spoof Lehmer factorizations with rplus positive factors and rminus negative factors.
    """
    # Our total number of factors
    r = rplus + rminus
    assert r > 1, "Our methods only apply if we have at least two factors"
    # We establish the bounds within which we need to work
    if base_spoof == None:
        lower_bound, upper_bound = partialSpoofLehmerFactorization(
            rplus, rminus, None, factors=None
        ).k_bounds()
    else:
        lower_bound, upper_bound = base_spoof.k_bounds()
    # For each of these bounds, we see what happens
    for k in range(ceil(lower_bound), floor(upper_bound) + 1):
        if verbose:
            print(f"k = {k}")
        if k != 1: # We should think more about this case and see if we can solve it in general! We're hackily avoiding the case n*(2 - n) right now
            for spoof in yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k(
                rplus, rminus, k, base_spoof = base_spoof, verbose=verbose
            ):
                yield spoof


def yield_all_spoof_Lehmer_factorizations_given_r(
    r: int,
    verbose: bool = True,
) -> Iterator[partialSpoofLehmerFactorization]:
    """
        Yields all interesting spoof Lehmer factorizations with given r.


    Parameters:
            r (int): The number of factors in our factorization.

    Yields:
        All nontrivial odd spoof Lehmer factorizations with r factors.
    """
    assert r > 1, "Our methods only apply if we have at least two factors"
    for rplus in range(0, r + 1):
        rminus = r - rplus
        if verbose:
            print(f"rplus = {rplus}, rminus = {rminus}")
        for spoof in yield_all_spoof_Lehmer_factorizations_given_rplus_rminus(rplus, rminus, verbose = verbose):
            yield spoof
