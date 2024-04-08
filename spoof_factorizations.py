from typing import List, Optional, Tuple
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

def product_of_list(numbers: List[int]) -> int:
    """
    Calculates the product of all integers in a given list.
    
    Parameters:
        numbers (List[int]): A list of integers whose product is to be found.
        
    Returns:
        int: The product of all integers in the list. Returns 1 for an empty list.
    
    Example:
        >>> product_of_list([1, 2, 3, 4])
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
        self.evaluation = product_of_list(self.factors)

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the partialSpoofLehmerFactorization instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"partialSpoofLehmerFactorization(rplus={self.rplus}, rminus={self.rminus}, k={self.k}, factors={self.factors})"

    def kUpper(self) -> Fraction:
        """
        Returns an upper bound on how large the ratio k(F) = F.evaluation/F.totient can be for F a (nontrivial odd) spoof factorization extending self compatible with our rplus and rminus conditions, and under the sorting asserted by sort_by_magnitude_then_positivity.
        """
        # If r- is odd, k(F) is increasing in its arguments so we let our negative terms be -|maximum magnitude of existing term| and our positive terms be infinity
        if self.rminus % 2 == 1:

        # If r- is even, k(F) is decreasing in its arguments so we let our negative terms be -infinity and our positive terms be |maximum magnitude of an existing term|

