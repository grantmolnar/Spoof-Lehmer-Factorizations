from typing import Iterator, List, Optional, Tuple
from fractions import Fraction
from functools import reduce
import operator
from collections import Counter

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
    assert all(isinstance(elem, (int, float, Fraction)) for elem in numbers)
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
    assert isinstance(output, int)
    return output


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

    Yields:
        int: The next integer in the sequence.
    """
    n = start
    if n <= 0:
        yield n
        n = abs(n) + step
    while True:
        yield n
        yield -n
        n += step


# Example usage:
iterator = integer_magnitude_iterator(-3)
for _ in range(10):  # Print first 10 values for demonstration
    print(next(iterator))

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
        assert self.splus <= self.rplus and self.sminus <= self.rminus
        # self.evaluation = compute_product_of_list(self.factors)
        # self.totient = compute_product_of_list([factor - 1 for factor in self.factors])

    def with_additional_factors(self, **kwargs): #-> partialSpoofLehmerFactorization
        """
        Produces a new instance of spoofLehmerFactorization with additional factors appended to the list of factors.
        
        Parameters:
            kwargs are new factors we can include
        
        Returns:
            spoofLehmerFactorization: A new instance of spoofLehmerFactorization with the updated list of factors.
        """
        extended_factors = self.factors.copy()
        extended_factors += [factor for factor in kwargs.values()]
        # Create a new instance with the same rplus, rminus, and k values, and the extended_factors list
        return partialSpoofLehmerFactorization(rplus=self.rplus, rminus=self.rminus, k=self.k, factors=extended_factors)

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
        # If r- is odd, k(F) is increasing in its arguments,
        # so we let our negative terms be -|maximum magnitude of existing term| and our positive terms be infinity to obtain an upper bound
        # and we let our negative terms be -infinity and our positive terms be |maximum magnitude of an existing term|
        # If r- is even, k(F) is decreasing in its arguments so we apply interchanged bounds
        # The number of new positive terms we will need to augment by
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
        negative_augmented_factors = self.factors + [negative_term] * new_negative_term_count
        # This is an upper bound if r- % 2 == 1, and a lower bound otherwise
        bound_1 = Fraction(evaluate(negative_augmented_factors) - (1 if new_positive_term_count == 0 else 0), compute_totient(negative_augmented_factors))
        positive_augmented_factors = self.factors + [positive_term] * new_positive_term_count
        # This is a lower bound if r- % 2 == 1, and an upper bound otherwise
        bound_2 = Fraction(
                evaluate(positive_augmented_factors)
                - (1 if new_negative_term_count == 0 else 0),
                compute_totient(positive_augmented_factors),
            )
        if self.rminus % 2 == 1:
            return (bound_2, bound_1)
        else:
            return (bound_1, bound_2)

def yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k(rplus : int, rminus: int, k : int, base_spoof : Optional[partialSpoofLehmerFactorization] = None)-> Iterator[partialSpoofLehmerFactorization]:
    """
    Returns an upper and lower bound on how large the ratio k(F) = F.evaluation/F.totient can be for F a (nontrivial odd) spoof factorization extending self compatible with our rplus and rminus conditions, and under the sorting asserted by sort_by_magnitude_then_positivity.

    Parameters:
            rplus (int): The positive component of the factorization.
            rminus (int): The negative component of the factorization.
            k (Optional[int]): An optional adjustment parameter, which we'd like to satisfy k * phi(F) = e(F) - 1
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.

    Yields:
        All nontrivial odd spoof Lehmer factorizations with rplus positive factors, rminus negative factors, and such that k = (evaluate(F) - 1)/(compute_totient(F))
    """
    # If our base spoof is none, we initialize an empty spoof
    if base_spoof == None:
        base_spoof = partialSpoofLehmerFactorization(rplus, rminus, k, factors = None)
    # If our base spoof is complete, we check if it works
    elif base_spoof.rplus == base_spoof.splus and base_spoof.rminus == base_spoof.sminus:
        if k*base_spoof.totient() == base_spoof.evaluation() - 1:
            yield base_spoof
    # Otherwise, our base_spoof is incomplete and we will augment it if we can
    else:
        upper_bound, lower_bound = base_spoof.k_bounds()
        # If our upper or lower bounds are incompatible with k, there is no need to go further.
        if lower_bound <= k and upper_bound <= k:
            # We need to consider the case of augmenting with new positive factors, and of augmenting with new negative factors, separately
            pass
